import random
import collections
import utils
import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
#from spinup_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
#from spinup_utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup_utils.logx import EpochLogger
from PIL import Image
import imageio
#from clip_model import build_model, tokenize_batch
#from torchvision.transforms import Resize
#from skimage.transform import resize
from mineclip_official import build_pretrain_model, tokenize_batch, torch_normalize
from minecraft import MinecraftEnv, preprocess_obs, transform_action
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import copy
from mineclip.utils import build_mlp


class Qnet(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        action: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        self.preprocess = preprocess_net
        self.net = build_mlp(
            input_dim=preprocess_net.output_dim,
            output_dim=action,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None,
        )
        self._action = action
        self._device = device

    def forward(self, x):
        y, _ = self.preprocess(x)
        return self.net(y)

class DQNbuffer:

    def __init__(self, act_dim, size=10000):
        self.obs_buf = [Batch() for i in range(size)]#np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_o_buf = [Batch() for i in range(size)]
        self.max_size = size
        self.full_flag = False
        self.ptr = 0

    def store(self, obs, act, rew, next_o):
        if self.ptr >= self.max_size:
            self.full_flag = True
            self.ptr = 0
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_o_buf[self.ptr] = next_o
        self.ptr += 1

    def sample(self, batch_size):
        if self.full_flag:
            get_id = random.sample(range(0,self.max_size),batch_size)
        else:
            get_id = random.sample(range(0,self.ptr),batch_size)
        data = dict(act=self.act_buf[get_id],rew=self.rew_buf[get_id])
        rtn = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        # rtn['obs'] = self.obs_buf[get_id]
        # rtn['next_o'] = self.next_o_buf[get_id]
        _obs = list()
        _next_obs = list()
        for i in get_id:
            _obs.append(self.obs_buf[i])
            _next_obs.append(self.next_o_buf[i])
        
        rtn['obs'] = Batch.cat(_obs)
        rtn['next_o'] = Batch.cat(_next_obs)

        return rtn

    def size(self):
        if self.full_flag:
            return self.max_size
        else:
            return self.ptr


# maintain the text embedding and images queue embedding
# compute constrastive intrinsic reward on the fly
class CLIPReward:
    def __init__(self, clip_model, device, text):
        self.clip_model = clip_model
        self.device = device
        self.text = text

        # load negative prompts
        with open('negative_prompts.txt', 'r') as f:
            self.neg_text = f.read().splitlines()
        # print(self.text, self.neg_text)

        # maintain an images embedding queue (16 frames)
        video = torch_normalize(np.zeros([1, 16, 3, 160, 256])).to(self.device)
        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float))
            # pre-compute text embedding
            self.text_emb = self.clip_model.text_encoder(tokenize_batch(self.text + self.neg_text).to(self.device))
            # print(tokenize_batch(self.text + self.neg_text), self.text_emb)
            assert self.text_emb.shape[0] == 1 + len(self.neg_text)

    # update the images queue when calling env.reset() or step()
    # the encoding for the new image is pre-computed in the env wrapper
    def update_obs(self, emb):
        assert emb.shape[0] == 1 and emb.shape[1] == 1  # (1,1,512)
        self.imgs_emb = torch.cat((self.imgs_emb[:, 1:], torch.as_tensor(emb, device=self.device)), 1)

    # compute the intrinsic reward at current state
    # mode: direct, direct-naive and delta in minedojo paper
    def reward(self, mode='direct'):
        with torch.no_grad():
            v_emb = self.clip_model.temporal_encoder(self.imgs_emb)
            adapted_video, adapted_text = self.clip_model.reward_adapter(v_emb, self.text_emb)
            v_f = adapted_video / adapted_video.norm(dim=1, keepdim=True)  # (1, 512)
            v_f = self.clip_model.logit_scale.exp() * v_f
            t_f = adapted_text / adapted_text.norm(dim=1, keepdim=True)  # (1, 512)
            # print(v_f.shape, t_f.shape)
            logits_per_video = v_f @ t_f.t()  # (1,32)
            prob = F.softmax(logits_per_video, dim=-1)[0][
                0].detach().cpu().numpy()  # P(video corresponds to the prompt)

        if mode == 'direct':
            assert self.text_emb.shape[0] == 32
            r_clip = max(prob - 1. / 32, 0)
        else:
            raise NotImplementedError
        return r_clip

    # reset the queue when env.reset
    def reset(self):
        video = torch_normalize(np.zeros([1, 16, 3, 160, 256])).to(self.device)
        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float))

def DQN(args, seed=0, device=None,
        steps_per_epoch=400, epochs=500, gamma=0.99, qf_lr=1e-4, epsilon=0.05, target_update=5, minimal_size = 105,
        action_dim=36, batch_size=100 ,max_ep_len=1000, save_freq=5, logger_kwargs=dict(), save_path='checkpoint',
        clip_config_path='', clip_model_path='', agent_config_path=''):
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load pretrained mineclip model
    clip_config = utils.get_yaml_data(clip_config_path)
    model_clip = build_pretrain_model(
        image_config=clip_config['image_config'],
        text_config=clip_config['text_config'],
        temporal_config=clip_config['temporal_config'],
        adapter_config=clip_config['adaptor_config'],
        state_dict=torch.load(clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model loaded.')

    # Instantiate environment
    env = MinecraftEnv(
        task_id=args.task,
        image_size=(160, 256),
        max_step=args.horizon,
        clip_model=model_clip,
        device=device,
        seed=seed,
        dense_reward=bool(args.use_dense)
    )
    obs_dim = env.observation_size
    env_act_dim = env.action_size
    agent_act_dim = len(args.actor_out_dim)
    print('Task prompt:', env.task_prompt)

    # Create DQN agent
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
    #feature_net_v = copy.deepcopy(feature_net)  # actor and critic do not share
    # #feature_net finish
    dqn = Qnet(
        feature_net,
        action=action_dim,  # [3, 3, 4, 25, 25, 8],
        device=device,
        **agent_config['actor'],
    ).to(device)
    dqn.eval()
    target_net = copy.deepcopy(dqn)
    optimizer = torch.optim.Adam(dqn.parameters(), lr= qf_lr)
    # Count variables
    var_counts = (utils.count_vars(dqn), utils.count_vars(model_clip))
    logger.log('\nNumber of parameters: \t dqn: %d, \t mineclip: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = DQNbuffer(agent_act_dim)

    def take_action(obs, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(0,action_dim)
        else:
            action = dqn(obs.obs).argmax().item()
        act = action_process(action)
        return act

    def action_process(act):
        action = torch.zeros((1,2), dtype=int)
        action[0][1] = act % 3
        action[0][0] = act // 3
        return action

    local_target_update = target_update
    local_update_count = 0

    def update(data, update_count):
        obs,act,rew,next_o = data['obs'],data['act'].to(device), \
                            data['rew'].to(device),data['next_o']
        obs.to_torch(device=device)
        next_o.to_torch(device=device)
        act = act.to(torch.int64)
        act = act.to(device=device)
        # print(act.device)
        rew = rew.unsqueeze(1)
        # print(rew.shape)
        _act = (torch.tensor([batch_size,1],dtype=torch.int64)).to(device=device)
        _act = (act[:,0]*3 + act[:,1]).unsqueeze(1)
        #print(torch.index_select(act, 1,torch.tensor([0])).device)
        # print(_act)
        q_values = dqn(obs.obs).gather(1, _act)
        max_next_q_values = target_net(next_o.obs).max(1)[0].view(-1,1)
        # print(max_next_q_values.shape)
        
        q_targets = rew + gamma * max_next_q_values
        # print(q_values)
        # print(q_targets)
        # print(q_targets.shape)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        optimizer.zero_grad()
        dqn_loss.backward()
        optimizer.step()
        update_count += 1
        if update_count % local_target_update == 0:
            target_net.load_state_dict(dqn.state_dict())  # 更新目标网络

    start_time = time.time()

    # initialize the clip reward model
    clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        # save a video of test
        def test_video():
            pth = os.path.join(save_path, '{}.gif'.format(epoch))
            # if not os.path.exists(pth):
            #    os.mkdir(pth)
            dqn.eval()  # in eval mode, the actor is also stochastic now
            obs = env.reset()
            gameover = False
            # i = 0
            img_list = []
            while True:
                img_list.append(np.transpose(obs['rgb'], [1, 2, 0]).astype(np.uint8))
                if gameover:
                    break
                batch = preprocess_obs(obs, device)
                act = take_action(batch,epsilon)
                act = transform_action(act)
                obs, r, gameover, _ = env.step(act)
                # i += 1
            imageio.mimsave(pth, img_list, duration=0.1)
            print("save success")
            # env.reset()
            # mine_agent.train()

        # Save model and test
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            test_video()
            # logger.save_state({'env': env}, None)
            pth = os.path.join(save_path, 'model_{}.pth'.format(epoch))
            torch.save(dqn.state_dict(), pth)

        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0  # Prepare for interaction with environment
        clip_reward_model.update_obs(o['rgb_emb'])  # preprocess the images embedding
        ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
        #rgb_list = []
        # rollout in the environment
        dqn.train()  # train mode to sample stochastic actions
        target_net.train()
        for t in range(local_steps_per_epoch):
            #if args.save_raw_rgb:
            #    rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))

            batch_o = preprocess_obs(o, device)
            batch_act = take_action(batch_o,epsilon)
            a = batch_act
            # print('a,v,logp = ', a, v, logp)

            a_env = transform_action(a)
            next_o, r, d, _ = env.step(a_env)
            success = r
            batch_next_o = preprocess_obs(next_o, device)
            # update the recent 16 frames, compute intrinsic reward
            clip_reward_model.update_obs(next_o['rgb_emb'])
            r_clip = clip_reward_model.reward(mode=args.clip_reward_mode)

            r = r * args.reward_success + r_clip * args.reward_clip + args.reward_step  # weighted sum of different rewards

            # dense reward
            if args.use_dense:
                r_dense = next_o['dense_reward']
                r += r_dense * args.reward_dense
                ep_ret_dense += r_dense

            ep_success += success
            ep_ret_clip += r_clip
            ep_ret += r
            ep_len += 1

            # save and log
            batch_o.to_numpy()  # less gpu mem
            batch_next_o.to_numpy()
            buf.store(batch_o, a[0].cpu(), r, batch_next_o)

            # Update obs (critical!)
            o = next_o

            if buf.size() > minimal_size:
                data = buf.sample(batch_size)
                update(data,local_update_count)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # check and add to imitation buffer if the trajectory ends
                # if terminal:
                    # obs_ = Batch.cat(buf.obs_buf[buf.path_start_idx: buf.ptr])
                    # act_ = buf.act_buf[buf.path_start_idx: buf.ptr].copy()
                    # rgb_list = np.asarray(rgb_list)
                    # print(rgb_list.shape)
                    # imitation_buf.eval_and_store(obs_, act_, ep_ret_clip, int(ep_success), rgb_list)

                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpSuccess=ep_success,
                                 EpRetDense=ep_ret_dense, EpRetClip=ep_ret_clip)

                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
                clip_reward_model.reset()  # don't forget to reset the clip images buffer
                clip_reward_model.update_obs(o['rgb_emb'])  # preprocess the images embedding
                #rgb_list = []

            # to avoid destroying too many blocks, remake the environment
            if (epoch % 50 == 0) and epoch > 0:
                env.remake_env()
                # save the imitation learning buffer
                pth = os.path.join(save_path, 'buffer_{}.pth'.format(epoch))

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRetClip', with_min_and_max=True)
        logger.log_tabular('EpSuccess', with_min_and_max=True)
        logger.log_tabular('EpRetDense', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()




