import utils
import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from spinup_utils.logx import EpochLogger
from PIL import Image
import imageio
from mineclip_official import build_pretrain_model, tokenize_batch, torch_normalize
from minecraft import MinecraftEnv, preprocess_obs, transform_action
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import copy
import pickle


class GRPOBuffer:
    """
    GRPO Change:
    A buffer for storing trajectories experienced by a GRPO agent.
    """

    def __init__(self, act_dim, size=1000, gamma=0.99, agent_model='mineagent', obs_dim=None):
        self.agent_model = agent_model
        if agent_model == 'mineagent':
            self.obs_buf = [Batch() for i in range(size)]
        else:
            self.obs_buf = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma # Gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.traj_boundaries = []

    def store(self, obs, act, rew, logp):
        assert self.ptr < self.max_size    # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def modify_trajectory_rewards(self, rews):
        assert self.ptr - self.path_start_idx == len(rews)
        self.rew_buf[self.path_start_idx: self.ptr] = rews

    def finish_path(self):
        path_slice = slice(self.path_start_idx, self.ptr)
        self.traj_boundaries.append(path_slice)
        self.path_start_idx = self.ptr

    def get(self):
        """
        [Modification 1: Return-to-Go based Credit Assignment]
        기존: 트래젝토리 총 보상 합(Sum)을 모든 스텝에 동일 적용
        수정: 각 스텝별 Return-to-Go (Discounted Cumulative Sum) 적용
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # 1. Calculate Return-to-Go for each step in each trajectory
        for path_slice in self.traj_boundaries:
            rews = self.rew_buf[path_slice]

            # 핵심 수정: 뒤에서부터 누적 합 (Return-to-Go) 계산
            # G_t = r_t + gamma * r_{t+1} + ...
            ret_go = utils.discount_cumsum(rews, self.gamma)

            # Advantage 버퍼에 Return-to-Go 저장
            self.adv_buf[path_slice] = ret_go

        # 2. Normalize across the whole group (batch)
        # GRPO의 핵심: 전체 배치 내에서의 상대적 우위 계산
        vals = self.adv_buf[:self.max_size]
        mean_adv = np.mean(vals)
        std_adv = np.std(vals) + 1e-8

        self.adv_buf = (self.adv_buf - mean_adv) / std_adv

        # 3. Reset boundaries
        self.traj_boundaries = []

        if self.agent_model == 'mineagent':
            data = dict(act=self.act_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn =  {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
            rtn['obs'] = Batch.cat(self.obs_buf)
        else:
            data = dict(obs=self.obs_buf, act=self.act_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        return rtn


# (SelfImitationBuffer class is unchanged)
class SelfImitationBuffer:
    def __init__(self, act_dim, size=500, imitate_success_only=True, agent_model='mineagent'):
        self.obs_buf = []
        self.act_buf = []
        self.ret_buf = []
        self.success_buf = []
        self.cur_size, self.max_size = 0, size
        self.baseline = 0.
        self.success_rate = 0.
        self.avg_return = 0.
        self.imitate_success_only = imitate_success_only
        self.i_saved_traj = 0
        self.agent_model = agent_model

    def eval_and_store(self, obs, act, ret, success, rgb=None, save_dir=None):
        if self.cur_size > 0:
            self.baseline = np.mean(self.ret_buf) + 2*np.std(self.ret_buf)
        if success or ((not self.imitate_success_only) and (ret >= self.baseline)):
            assert self.cur_size <= self.max_size
            self.obs_buf.append(obs)
            self.act_buf.append(act)
            self.ret_buf.append(ret)
            self.success_buf.append(success)
            self.success_rate = np.mean(self.success_buf)
            self.avg_return = np.mean(self.ret_buf)

            if self.cur_size < self.max_size:
                self.cur_size += 1
            else: # FIFO
                del(self.obs_buf[0])
                del(self.act_buf[0])
                del(self.ret_buf[0])
                del(self.success_buf[0])

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                pth = os.path.join(save_dir, 'traj_{}.pth'.format(self.i_saved_traj))
                pickle.dump([obs, act, ret, success, rgb], open(pth, 'wb'))
                self.i_saved_traj += 1

    def get(self):
        assert self.cur_size > 0
        act_ = np.concatenate(self.act_buf)
        if self.agent_model == 'mineagent':
            obs_ = Batch.cat(self.obs_buf)
            rtn = {
                'act': torch.as_tensor(act_, dtype=torch.long),
                'obs': obs_
            }
        else:
            obs_ = np.concatenate(self.obs_buf)
            rtn = {
                'act': torch.as_tensor(act_, dtype=torch.long),
                'obs': torch.as_tensor(obs_, dtype=torch.float32)
            }
        return rtn


# (CLIPReward class is unchanged)
class CLIPReward:
    def __init__(self, clip_model, device, text):
        self.clip_model = clip_model
        self.device = device
        self.text = text
        with open('negative_prompts.txt', 'r') as f:
            self.neg_text = f.read().splitlines()
        with torch.no_grad():
            self.text_emb = self.clip_model.text_encoder(tokenize_batch(self.text + self.neg_text).to(self.device))
            assert self.text_emb.shape[0] == 1+len(self.neg_text)

    def compute_all_embeddings(self, imgs):
        video_begin = torch_normalize(np.zeros([1, 15, 3, 160, 256])).to(self.device)
        video = imgs.to(self.device)
        video = torch.cat((video_begin, video), 1)
        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float))

    def reward(self, imgs_emb_window, mode='direct'):
        with torch.no_grad():
            v_emb = self.clip_model.temporal_encoder(imgs_emb_window)
            adapted_video, adapted_text = self.clip_model.reward_adapter(v_emb, self.text_emb)
            v_f = adapted_video / adapted_video.norm(dim=1, keepdim=True)
            v_f = self.clip_model.logit_scale.exp()*v_f
            t_f= adapted_text / adapted_text.norm(dim=1, keepdim=True)
            logits_per_video = v_f @ t_f.t()
            prob = F.softmax(logits_per_video, dim=-1)[0][0].detach().cpu().numpy()

        if mode=='direct':
            assert self.text_emb.shape[0] == 32
            r_clip = max(prob - 1./32, 0)
        else:
            raise NotImplementedError
        return r_clip

    def compute_all_rewards(self, mode='direct'):
        r = []
        for i in range(self.imgs_emb.shape[1]-16):
            r.append(self.reward(self.imgs_emb[:, i+1:i+17], mode=mode))
        return np.array(r)


def grpo_selfimitate_clip(args, seed=0, device=None,
        steps_per_epoch=400, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4,
        train_pi_iters=80, max_ep_len=1000,
        target_kl=0.01, save_freq=5, logger_kwargs=dict(), save_path='checkpoint',
        clip_config_path='', clip_model_path='', agent_config_path=''):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load pretrained mineclip model
    clip_config = utils.get_yaml_data(clip_config_path)
    model_clip = build_pretrain_model(
        image_config = clip_config['image_config'],
        text_config = clip_config['text_config'],
        temporal_config = clip_config['temporal_config'],
        adapter_config = clip_config['adaptor_config'],
        state_dict = torch.load(clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model loaded.')

    # Instantiate environment
    env = MinecraftEnv(
        task_id=args.task,
        image_size=(160, 256),
        max_step=args.horizon,
        clip_model=model_clip if (args.agent_model == 'mineagent') else None,
        device=device,
        seed=seed,
        dense_reward=bool(args.use_dense)
    )
    obs_dim = env.observation_size
    env_act_dim = env.action_size
    agent_act_dim = len(args.actor_out_dim)
    print('Task prompt:', env.task_prompt)

    # Create actor
    if args.agent_model == 'mineagent':
        agent_config = utils.get_yaml_data(agent_config_path)
        feature_net_kwargs = agent_config['feature_net_kwargs']
        feature_net = {}
        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(features, cls)
            feature_net[k] = cls(**v, device=device)
        feature_fusion_kwargs = agent_config['feature_fusion']
        feature_net = SimpleFeatureFusion(
            feature_net, **feature_fusion_kwargs, device=device
        )
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=args.actor_out_dim,
            device=device,
            **agent_config['actor'],
            activation='tanh',
        )
        mine_agent = MineAgent(
            actor=actor,
            critic=None,
            deterministic_eval=False
        ).to(device)
        mine_agent.eval()
    elif args.agent_model == 'cnn':
        mine_agent = utils.CNNActorCritic(
            action_dim=args.actor_out_dim,
            deterministic_eval=False,
        ).to(device)
        mine_agent.eval()
    else:
        raise NotImplementedError

    # Count variables
    var_counts = (utils.count_vars(mine_agent.actor), utils.count_vars(model_clip))
    logger.log('\nNumber of parameters: \t actor: %d, \t mineclip: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = GRPOBuffer(agent_act_dim, local_steps_per_epoch, gamma, args.agent_model, obs_dim)

    # set up imitation buffer
    imitation_buf = SelfImitationBuffer(agent_act_dim, args.imitate_buf_size, args.imitate_success_only, args.agent_model)


    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'].to(device), \
                                  data['adv'].to(device), data['logp'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)

        # Policy loss
        pi = mine_agent(obs).dist
        logp = pi.log_prob(act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    pi_optimizer = torch.optim.Adam(mine_agent.actor.parameters(), lr=pi_lr)

    # [Original update function - Full Batch]
    # Reverted Mini-batch modification as per user request.
    def update():
        mine_agent.train()
        data = buf.get() # dict

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']

            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl, kl=%f.'%(i, kl))
                break
            loss_pi.backward()
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old))


    def compute_loss_imitation(data, idxs):
        obs, act = data['obs'][idxs], data['act'][idxs].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)
        pi = mine_agent(obs).dist
        loss_imitation = pi.imitation_loss(act)
        return loss_imitation

    def update_imitation():
        mine_agent.train()
        data = imitation_buf.get()
        n_data = data['act'].shape[0]
        n_iter = max(int(n_data / args.imitate_batch_size), 1)
        for i in range(n_iter):
            pi_optimizer.zero_grad()
            idxs = np.random.randint(0, n_data, size=args.imitate_batch_size)
            loss_imitation = compute_loss_imitation(data, idxs)
            loss_imitation.backward()
            pi_optimizer.step()
        logger.store(LossImitation=loss_imitation.item(), NumItersImitation=n_iter)


    start_time = time.time()
    saved_traj_cnt = 0
    clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])


    # Main loop
    for epoch in range(epochs):

        # [Modification 2: Curriculum Learning]
        # Dynamically adjust dense reward weight based on epoch progress.
        # Linear Decay: Start at args.reward_dense, end at 0.
        # You can tune 'decay_end_epoch_ratio' (e.g., stop using dense reward after 50% of epochs)
        decay_end_epoch_ratio = 0.5
        progress = min(1.0, epoch / (epochs * decay_end_epoch_ratio))

        # Current Dense Reward Weight: linearly decays to 0
        cur_dense_weight = args.reward_dense * (1.0 - progress)

        # Current Success Reward Weight: stays same or increases (Here we keep it constant or you can increase it)
        cur_success_weight = args.reward_success

        # Logger for curriculum status
        if epoch % save_freq == 0:
            logger.log(f'[Curriculum] Epoch {epoch}: Dense Weight {cur_dense_weight:.4f} (Base {args.reward_dense})')

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            pth = os.path.join(save_path, 'model', 'model_{}.pth'.format(epoch))
            torch.save(mine_agent.state_dict(), pth)

        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0
        ep_rewards = []
        ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
        ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
        rgb_list = []
        episode_in_epoch_cnt = 0

        mine_agent.train()
        for t in range(local_steps_per_epoch):
            if args.save_raw_rgb:
                rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))

            if args.agent_model == 'mineagent':
                batch_o = preprocess_obs(o, device)
            else:
                batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,*obs_dim)
                batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)

            with torch.no_grad():
                batch_act = mine_agent.forward_actor_critic(batch_o)
                a, logp = batch_act.act, batch_act.logp
                logp = logp[0]

            a_env = transform_action(a)
            next_o, r, d, _ = env.step(a_env)
            success = r

            # Apply Curriculum Weight for success/step rewards if needed (kept original logic here)
            r = r * cur_success_weight + args.reward_step
            ep_rewards.append(r)
            ep_obs = torch.cat((ep_obs,
                torch_normalize(np.asarray(next_o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)), 1)

            if args.use_dense:
                r_dense = next_o['dense_reward']
                # [Curriculum Applied Here] Use calculated cur_dense_weight
                r += r_dense * cur_dense_weight
                ep_ret_dense += r_dense

            ep_success += success
            if ep_success > 1:
                ep_success = 1
            ep_ret += r
            ep_len += 1

            if args.agent_model == 'mineagent':
                batch_o.to_numpy()
            else:
                batch_o = batch_o.cpu().numpy()

            buf.store(batch_o, a[0].cpu().numpy(), r, logp)

            o = next_o
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                clip_reward_model.compute_all_embeddings(ep_obs)
                ep_rewards_clip = clip_reward_model.compute_all_rewards()
                ep_rewards = np.asarray(ep_rewards) + args.reward_clip * ep_rewards_clip
                ep_ret_clip = np.sum(ep_rewards_clip)
                ep_ret += ep_ret_clip
                buf.modify_trajectory_rewards(ep_rewards)

                if terminal:
                    if args.agent_model == 'mineagent':
                        obs_ = Batch.cat(buf.obs_buf[buf.path_start_idx: buf.ptr])
                    else:
                        obs_ = buf.obs_buf[buf.path_start_idx: buf.ptr].copy()
                    act_ = buf.act_buf[buf.path_start_idx: buf.ptr].copy()
                    if args.save_raw_rgb:
                        rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
                    rgb_list = np.asarray(rgb_list)
                    expert_save_dir = os.path.join(args.save_path, 'expert_buffer') if args.save_expert_data else None
                    imitation_buf.eval_and_store(obs_, act_, ep_ret_clip, int(ep_success), rgb_list, expert_save_dir)

                    if args.save_all_data:
                        pth = os.path.join(args.save_path, 'experience_buffer', 'traj_{}.pth'.format(saved_traj_cnt))
                        pickle.dump([obs_, act_, ep_ret_clip, int(ep_success), rgb_list], open(pth, 'wb'))
                        saved_traj_cnt += 1

                    if args.save_raw_rgb and ((epoch % save_freq == 0) or (epoch == epochs-1)) and episode_in_epoch_cnt==0:
                        pth = os.path.join(args.save_path, 'gif', '{}_ret{}_success{}.gif'.format(epoch, int(ep_ret), int(ep_success)))
                        imageio.mimsave(pth, [np.transpose(i_, [1,2,0]) for i_ in rgb_list], duration=0.1)

                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

                buf.finish_path()
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRetClip=ep_ret_clip, EpSuccess=ep_success,
                        EpRetDense=ep_ret_dense)

                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
                ep_rewards = []
                ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
                rgb_list = []
                episode_in_epoch_cnt += 1


        # Perform GRPO update (Original Full Batch)
        update()
        episode_in_epoch_cnt = 0

        # Perform self-imitation
        if imitation_buf.cur_size >= 1 and (epoch % args.imitate_freq == 0) and epoch > 0:
            for i_imitate in range(args.imitate_epoch):
                update_imitation()
            logger.store(ImitationBufferSuccess=imitation_buf.success_rate,
                ImitationBufferReturn=imitation_buf.avg_return,
                ImitationBufferAcceptReturn=imitation_buf.baseline,
                ImitationBufferNumTraj=imitation_buf.cur_size)
            logger.log_tabular('LossImitation', average_only=True)
            logger.log_tabular('NumItersImitation', average_only=True)
            logger.log_tabular('ImitationBufferSuccess', average_only=True)
            logger.log_tabular('ImitationBufferReturn', average_only=True)
            logger.log_tabular('ImitationBufferAcceptReturn', average_only=True)
            logger.log_tabular('ImitationBufferNumTraj', average_only=True)
        elif epoch == 0:
            logger.log_tabular('LossImitation', 0)
            logger.log_tabular('NumItersImitation', 0)
            logger.log_tabular('ImitationBufferSuccess', 0)
            logger.log_tabular('ImitationBufferReturn', 0)
            logger.log_tabular('ImitationBufferAcceptReturn', 0)
            logger.log_tabular('ImitationBufferNumTraj', 0)


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRetClip', with_min_and_max=True)
        logger.log_tabular('EpSuccess', with_min_and_max=True)
        logger.log_tabular('EpRetDense', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        if (epoch % 50 == 0) and epoch>0:
            env.remake_env()
            pth = os.path.join(save_path, 'newbuffer_{}.pth'.format(epoch))
            pickle.dump(imitation_buf, open(pth, 'wb'))
