import utils
import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
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
import pickle


# GRPO Change: PPOBuffer -> GRPOBuffer
# for mineagent: observation is stored with Batch
# for CNN actor: observation should be processed by torch_normalize before save
class GRPOBuffer:
    """
    GRPO Change:
    A buffer for storing trajectories experienced by a GRPO agent.
    Instead of GAE, it computes relative advantages based on the
    total rewards of all trajectories (the "group") within the buffer.
    """

    # GRPO Change: Removed lam, obs_dim (obs_dim wasn't used for mineagent anyway)
    def __init__(self, act_dim, size=1000, gamma=0.99, agent_model='mineagent', obs_dim=None):
        self.agent_model = agent_model
        if agent_model == 'mineagent':
            self.obs_buf = [Batch() for i in range(size)]
        else:
            self.obs_buf = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # GRPO Change: Removed ret_buf (Return-to-go) and val_buf (Value)
        # self.ret_buf = np.zeros(size, dtype=np.float32)
        # self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # GRPO Change: Removed lam
        self.gamma = gamma # Gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        # GRPO Change: Add list to store trajectory boundaries
        self.traj_boundaries = []

    # GRPO Change: Removed 'val' from parameters
    def store(self, obs, act, rew, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size    # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # GRPO Change: Removed val_buf
        # self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def modify_trajectory_rewards(self, rews):
        """
        modify the recently saved rewards with a numpy array: rews.
        should be called after many store(), and before finish_path().
        """
        assert self.ptr - self.path_start_idx == len(rews)
        self.rew_buf[self.path_start_idx: self.ptr] = rews

    # GRPO Change: Removed last_val. This function no longer computes GAE.
    # It just records the boundary of the completed trajectory.
    def finish_path(self):
        """
        GRPO Change:
        Call this at the end of a trajectory. This function no longer
        computes GAE or rewards-to-go. It simply records the
        trajectory's boundary (start and end index) for later
        computation of relative rewards in the get() method.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        # Store the boundary of this trajectory
        self.traj_boundaries.append(path_slice)
        self.path_start_idx = self.ptr

    def get(self):
        """
        GRPO Change:
        Call this at the end of an epoch to get all of the data from
        the buffer.
        This function now computes the relative advantage for each
        trajectory in the "group" (all trajectories in the buffer).
        A_i = (R_i - mean(R_group)) / std(R_group)
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # GRPO Change: Compute relative rewards (Advantage)
        # 1. Calculate total reward (R_i) for each trajectory
        group_total_rewards = []
        for path_slice in self.traj_boundaries:
            # Note: Using simple sum of rewards (R_i) as per the GRPO formula
            # If discounted return is desired, use utils.discount_cumsum(self.rew_buf[path_slice], self.gamma)[0]
            R_i = np.sum(self.rew_buf[path_slice])
            group_total_rewards.append(R_i)

        group_total_rewards = np.array(group_total_rewards)

        # 2. Calculate mean and std of the group's rewards
        mean_R = np.mean(group_total_rewards)
        std_R = np.std(group_total_rewards) + 1e-8 # Add epsilon to avoid division by zero

        # 3. Calculate relative advantage (A_i) for each trajectory
        traj_advantages = (group_total_rewards - mean_R) / std_R

        # 4. Assign this constant advantage to all steps in the corresponding trajectory
        for i, path_slice in enumerate(self.traj_boundaries):
            self.adv_buf[path_slice] = traj_advantages[i]

        # 5. Reset trajectory boundaries for the next epoch
        self.traj_boundaries = []

        # the next two lines implement the advantage normalization trick (still useful)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std


        if self.agent_model == 'mineagent':
            # GRPO Change: Removed 'ret' from data dictionary
            data = dict(act=self.act_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn =  {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
            rtn['obs'] = Batch.cat(self.obs_buf)
        else:
            # GRPO Change: Removed 'ret' from data dictionary
            data = dict(obs=self.obs_buf, act=self.act_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        return rtn


# (SelfImitationBuffer class is unchanged)
class SelfImitationBuffer:
    # ... (no changes in this class) ...
    def __init__(self, act_dim, size=500, imitate_success_only=True, agent_model='mineagent'):
        '''
        each saved item is a trajectory: act_buf [[len, act_dim], ...]
        '''
        self.obs_buf = []
        self.act_buf = [] # np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        self.ret_buf = [] # returns
        self.success_buf = []
        self.cur_size, self.max_size = 0, size
        self.baseline = 0.
        self.success_rate = 0.
        self.avg_return = 0.
        self.imitate_success_only = imitate_success_only

        #self.rgb_buf = []
        self.i_saved_traj = 0
        self.agent_model = agent_model

    # eval the trajectory performance and decide to store
    def eval_and_store(self, obs, act, ret, success, rgb=None, save_dir=None):
        '''
        store if success or episode return >= baseline
        if the buffer is full, first-in-first-out
        '''
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
            #self.rgb_buf.append(rgb)

            if self.cur_size < self.max_size:
                self.cur_size += 1
            else: # FIFO
                del(self.obs_buf[0])
                del(self.act_buf[0])
                del(self.ret_buf[0])
                del(self.success_buf[0])
                #del(self.rgb_buf[0])

            # save the expert trajectory
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                pth = os.path.join(save_dir, 'traj_{}.pth'.format(self.i_saved_traj))
                pickle.dump([obs, act, ret, success, rgb], open(pth, 'wb'))
                self.i_saved_traj += 1


        #print(self.cur_size, len(self.obs_buf), self.baseline, self.success_rate,
        #    obs.shape, act.shape, ret)

    # get all the data for training.
    # convert the trajectory list [N * [len, dim]] to transition array [N', dim]
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
    # ... (no changes in this class) ...
    def __init__(self, clip_model, device, text):
        self.clip_model = clip_model
        self.device = device
        self.text = text

        # load negative prompts
        with open('negative_prompts.txt', 'r') as f:
            self.neg_text = f.read().splitlines()
        #print(self.text, self.neg_text)

        # create initial 15 empty frames before env reset
        #video = torch_normalize(np.zeros([1, 15, 3, 160, 256])).to(self.device)
        with torch.no_grad():
            #self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float))
            # pre-compute text embedding
            self.text_emb = self.clip_model.text_encoder(tokenize_batch(self.text + self.neg_text).to(self.device))
            #print(tokenize_batch(self.text + self.neg_text), self.text_emb)
            assert self.text_emb.shape[0] == 1+len(self.neg_text)

    '''
    # update the images queue when calling env.reset() or step()
    # the encoding for the new image is pre-computed in the env wrapper
    def update_obs(self,emb):
        assert emb.shape[0] == 1 and emb.shape[1] == 1 # (1,1,512)
        self.imgs_emb = torch.cat((self.imgs_emb[:,1:], torch.as_tensor(emb, device=self.device)), 1)
    '''

    # compute all the embeddings for a trajectory, concat the 15 empty frames at beginning
    # input imgs should be preprocessed by torch_normalize
    def compute_all_embeddings(self, imgs):
        #assert imgs.dtype is np.int
        video_begin = torch_normalize(np.zeros([1, 15, 3, 160, 256])).to(self.device) # pad 15 frames before reset
        video = imgs.to(self.device) # (1, N, 3, 160, 256)
        video = torch.cat((video_begin, video), 1) # (1, 15+N, 3, 160, 256)
        #print(video.shape)

        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float)) # (1, 15+N, 512)
            #print(self.imgs_emb.shape)


    # compute the intrinsic reward for a 16-frames window
    # mode: direct, direct-naive and delta in minedojo paper
    def reward(self, imgs_emb_window, mode='direct'):
        with torch.no_grad():
            v_emb = self.clip_model.temporal_encoder(imgs_emb_window)
            adapted_video, adapted_text = self.clip_model.reward_adapter(v_emb, self.text_emb)
            v_f = adapted_video / adapted_video.norm(dim=1, keepdim=True) # (1, 512)
            v_f = self.clip_model.logit_scale.exp()*v_f
            t_f= adapted_text / adapted_text.norm(dim=1, keepdim=True) # (1, 512)
            #print(v_f.shape, t_f.shape)
            logits_per_video = v_f @ t_f.t() # (1,32)
            prob = F.softmax(logits_per_video, dim=-1)[0][0].detach().cpu().numpy() # P(video corresponds to the prompt)

        if mode=='direct':
            assert self.text_emb.shape[0] == 32
            r_clip = max(prob - 1./32, 0)
        else:
            raise NotImplementedError
        return r_clip

    # compute reward sequence for a trajectory, after calling compute_all_embeddings
    def compute_all_rewards(self, mode='direct'):
        r = []
        for i in range(self.imgs_emb.shape[1]-16):
            r.append(self.reward(self.imgs_emb[:, i+1:i+17], mode=mode)) # skip the reset step
        return np.array(r)

    '''
    # reset the queue when env.reset
    def reset(self):
        video = torch_normalize(np.zeros([1, 16, 3, 160, 256])).to(self.device)
        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float))
    '''



'''
GRPO Change: PPO algorithm implementation -> GRPO algorithm implementation
for every epoch, first play the game to collect trajectories
until the buffer is full, then update the actor for sevaral steps using the buffer.

grpo_clip uses mineclip intrinsic reward for sparse reward tasks.
'''
# GRPO Change: Renamed function
# GRPO Change: Removed vf_lr, train_v_iters, lam from parameters
def grpo_selfimitate_clip(args, seed=0, device=None,
        steps_per_epoch=400, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4, # vf_lr=1e-4,
        train_pi_iters=80, max_ep_len=1000, # train_v_iters=80, lam=0.95,
        target_kl=0.01, save_freq=5, logger_kwargs=dict(), save_path='checkpoint',
        clip_config_path='', clip_model_path='', agent_config_path=''):

    """
    GRPO Change: Updated Docstring
    Group Relative Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        ( ... docstring describing env_fn, actor_critic ... )

        GRPO Change: Removed v (value) from step method docstring
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            ( ... docstring describing pi module ... )

        GRPO Change: Removed v module (Critic) from docstring
            ( ... )

        ( ... )

        pi_lr (float): Learning rate for policy optimizer.

        GRPO Change: Removed vf_lr
        ( ... )

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        GRPO Change: Removed train_v_iters
        ( ... )

        GRPO Change: Removed lam
        ( ... )
    """


    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    #setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())


    # Random seed
    #seed += 10000 * proc_id()
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
    #logger.log('env: obs {}, act {}'.format(env.observation_space, env.action_space))


    # Create actor-critic agent
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
        # GRPO Change: Removed feature_net_v (for critic)
        # feature_net_v = copy.deepcopy(feature_net) # actor and critic do not share
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=args.actor_out_dim, #[3, 3, 4, 25, 25, 8],
            device=device,
            **agent_config['actor'],
            activation='tanh',
        )
        # GRPO Change: Removed critic
        # critic = Critic(
        #     feature_net_v,
        #     action_dim=None,
        #     device=device,
        #     **agent_config['actor'],
        #     activation='tanh'
        # )
        mine_agent = MineAgent(
            actor=actor,
            # GRPO Change: Pass critic=None
            critic=None,
            deterministic_eval=False
        ).to(device) # use the same stochastic policy in training and test
        mine_agent.eval()
    elif args.agent_model == 'cnn':
        # GRPO Change: Assuming CNNActorCritic can handle critic=None or is modified
        # For this example, we'll assume it's modified to not need a critic
        # or we only use its actor part.
        # This part of the code might need adjustment based on CNNActorCritic implementation
        mine_agent = utils.CNNActorCritic(
            action_dim=args.actor_out_dim,
            deterministic_eval=False,
            # GRPO Change: Explicitly remove critic logic if possible
            # needs_critic=False # (Assuming such a parameter exists)
        ).to(device)
        mine_agent.eval()
    else:
        raise NotImplementedError

    # Sync params across processes
    #sync_params(ac)

    # Count variables
    var_counts = (#utils.count_vars(actor), # GRPO Change: removed critic
        utils.count_vars(mine_agent.actor), utils.count_vars(model_clip))
    #logger.log('\nNumber of parameters: \t actor: %d, \t critic: %d, \t  agent: %d, \t mineclip: %d\n'%var_counts)
    # GRPO Change: Updated log message
    logger.log('\nNumber of parameters: \t actor: %d, \t mineclip: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    # GRPO Change: Use GRPOBuffer, removed lam
    buf = GRPOBuffer(agent_act_dim, local_steps_per_epoch, gamma, args.agent_model, obs_dim)

    # set up imitation buffer
    imitation_buf = SelfImitationBuffer(agent_act_dim, args.imitate_buf_size, args.imitate_success_only, args.agent_model)


    # Set up function for computing PPO policy loss
    # GRPO Change: This is now the GRPO policy loss, but the formula is identical to PPO's clipped objective.
    def compute_loss_pi(data):
        # GRPO Change: 'ret' is no longer in data
        obs, act, adv, logp_old = data['obs'], data['act'].to(device), \
                                  data['adv'].to(device), data['logp'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)

        # Policy loss
        pi = mine_agent(obs).dist
        logp = pi.log_prob(act)
        #print('logp, logp_old = ', logp, logp_old)
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

    # GRPO Change: Removed compute_loss_v function entirely
    # def compute_loss_v(data):
    #     obs, ret = data['obs'], data['ret'].to(device)
    #     if args.agent_model == 'mineagent':
    #         obs.to_torch(device=device)
    #         obs_ = obs.obs
    #     else:
    #         obs_ = obs.to(device)
    #     return ((mine_agent.critic(obs_) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(mine_agent.actor.parameters(), lr=pi_lr)
    # GRPO Change: Removed vf_optimizer
    # vf_optimizer = torch.optim.Adam(mine_agent.critic.parameters(), lr=vf_lr)
    #optimizer = torch.optim.Adam(mine_agent.parameters(), lr=lr)


    # a training epoch
    def update():
        mine_agent.train()

        data = buf.get() # dict

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        # GRPO Change: Removed v_l_old
        # v_l_old = compute_loss_v(data).item()

        # GRPO Change: Removed Value function learning loop
        # for i in range(train_v_iters):
        #     vf_optimizer.zero_grad()
        #     loss_v = compute_loss_v(data)
        #     loss_v.backward()
        #     #mpi_avg_grads(ac.v)    # average grads across MPI processes
        #     vf_optimizer.step()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl'] #mpi_avg(pi_info['kl'])
            #logger.log('kl={}'.format(kl))
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl, kl=%f.'%(i, kl))
                break
            loss_pi.backward()
            #mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)


        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # GRPO Change: Removed LossV and DeltaLossV
        logger.store(LossPi=pi_l_old, # LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old))
                     #DeltaLossV=(loss_v.item() - v_l_old))



    # set up function for computing self-imitation loss
    # use the batch indexed by idxs in data
    def compute_loss_imitation(data, idxs):
        obs, act = data['obs'][idxs], data['act'][idxs].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)
        pi = mine_agent(obs).dist
        loss_imitation = pi.imitation_loss(act)
        return loss_imitation

    # training step for self-imitation learning
    def update_imitation():
        mine_agent.train()
        data = imitation_buf.get()
        n_data = data['act'].shape[0]
        n_iter = max(int(n_data / args.imitate_batch_size), 1) # iterations to train for 1 epoch
        #print('data', data, n_iter, n_data)
        for i in range(n_iter):
            pi_optimizer.zero_grad()
            idxs = np.random.randint(0, n_data, size=args.imitate_batch_size)
            #print('training batch', data['act'][idxs], data['obs'][idxs])
            loss_imitation = compute_loss_imitation(data, idxs)
            loss_imitation.backward()
            pi_optimizer.step()
        logger.store(LossImitation=loss_imitation.item(), NumItersImitation=n_iter)



    start_time = time.time()
    saved_traj_cnt = 0 # counter for the saved experience

    # initialize the clip reward model
    clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        '''
        # (Test video function unchanged)
        # save a video of test
        def test_video():
        ...
        '''

        # Save model and test
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            #test_video()
            #logger.save_state({'env': env}, None)
            pth = os.path.join(save_path, 'model', 'model_{}.pth'.format(epoch))
            torch.save(mine_agent.state_dict(), pth)


        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0 # Prepare for interaction with environment
        #clip_reward_model.update_obs(o['rgb_emb']) # preprocess the images embedding
        ep_rewards = []
        ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
        ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
        rgb_list = []
        episode_in_epoch_cnt = 0 # episode id in this epoch


        # rollout in the environment
        mine_agent.train() # train mode to sample stochastic actions
        for t in range(local_steps_per_epoch):
            if args.save_raw_rgb:
                rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))

            if args.agent_model == 'mineagent':
                batch_o = preprocess_obs(o, device)
            else:
                batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,*obs_dim)
                batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)

            with torch.no_grad():
                # GRPO Change: forward_actor_critic might still be the function name
                # but we will ignore the 'val' output.
                batch_act = mine_agent.forward_actor_critic(batch_o)
                # GRPO Change: We ignore 'v' (batch_act.val)
                a, logp = batch_act.act, batch_act.logp
                # v = v[0] # We don't need v
                logp = logp[0]
                #print('a,v,logp = ', a, v, logp)

            a_env = transform_action(a)
            next_o, r, d, _ = env.step(a_env)
            success = r

            # update the recent 16 frames, compute intrinsic reward
            #clip_reward_model.update_obs(next_o['rgb_emb'])
            #r_clip = clip_reward_model.reward(mode=args.clip_reward_mode)

            r = r * args.reward_success + args.reward_step # + r_clip * args.reward_clip # weighted sum of different rewards
            ep_rewards.append(r)
            ep_obs = torch.cat((ep_obs,
                torch_normalize(np.asarray(next_o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)), 1)

            # dense reward
            if args.use_dense:
                r_dense = next_o['dense_reward']
                r += r_dense * args.reward_dense
                ep_ret_dense += r_dense

            ep_success += success
            if ep_success > 1:
                ep_success = 1
            #ep_ret_clip += r_clip
            ep_ret += r
            ep_len += 1

            # save and log
            if args.agent_model == 'mineagent':
                batch_o.to_numpy() # less gpu mem
            else:
                batch_o = batch_o.cpu().numpy()
            # GRPO Change: Removed 'v' from buf.store() call
            buf.store(batch_o, a[0].cpu().numpy(), r, logp) # the stored reward will be modified at episode end, if use CLIP reward
            # GRPO Change: Removed VVals logging
            # logger.store(VVals=v.detach().cpu().numpy())

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                # compute CLIP embeddings and rewards for each step.
                # modify the trajectory rewards in the buffer
                clip_reward_model.compute_all_embeddings(ep_obs)
                ep_rewards_clip = clip_reward_model.compute_all_rewards()
                #print(len(ep_rewards_clip), len(ep_rewards), ep_obs.shape)
                ep_rewards = np.asarray(ep_rewards) + args.reward_clip * ep_rewards_clip
                ep_ret_clip = np.sum(ep_rewards_clip)
                ep_ret += ep_ret_clip # GRPO Change: This ep_ret is R_i
                buf.modify_trajectory_rewards(ep_rewards)

                # check and add to imitation buffer if the trajectory ends
                if terminal:
                    if args.agent_model == 'mineagent':
                        obs_ = Batch.cat(buf.obs_buf[buf.path_start_idx: buf.ptr])
                    else:
                        obs_ = buf.obs_buf[buf.path_start_idx: buf.ptr].copy()
                    act_ = buf.act_buf[buf.path_start_idx: buf.ptr].copy()
                    if args.save_raw_rgb:
                        rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
                    rgb_list = np.asarray(rgb_list)
                    #print(rgb_list.shape)
                    expert_save_dir = os.path.join(args.save_path, 'expert_buffer') if args.save_expert_data else None
                    imitation_buf.eval_and_store(obs_, act_, ep_ret_clip, int(ep_success), rgb_list, expert_save_dir)

                    # save the experience
                    if args.save_all_data:
                        pth = os.path.join(args.save_path, 'experience_buffer', 'traj_{}.pth'.format(saved_traj_cnt))
                        pickle.dump([obs_, act_, ep_ret_clip, int(ep_success), rgb_list], open(pth, 'wb'))
                        saved_traj_cnt += 1

                    # save the gif
                    if args.save_raw_rgb and ((epoch % save_freq == 0) or (epoch == epochs-1)) and episode_in_epoch_cnt==0:
                        pth = os.path.join(args.save_path, 'gif', '{}_ret{}_success{}.gif'.format(epoch, int(ep_ret), int(ep_success)))
                        imageio.mimsave(pth, [np.transpose(i_, [1,2,0]) for i_ in rgb_list], duration=0.1)


                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

                    # GRPO Change: Removed bootstrapping logic.
                    # GRPO does not use a value function.
                # if trajectory didn't reach terminal state, bootstrap value target
                # if timeout or epoch_ended:
                #     ... (removed v calculation) ...
                # else:
                #     v = 0

                # GRPO Change: Call finish_path without 'v'
                buf.finish_path()
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRetClip=ep_ret_clip, EpSuccess=ep_success,
                        EpRetDense=ep_ret_dense)

                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
                ep_rewards = []
                ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
                #clip_reward_model.reset() # don't forget to reset the clip images buffer
                #clip_reward_model.update_obs(o['rgb_emb']) # preprocess the images embedding
                rgb_list = []
                episode_in_epoch_cnt += 1


        # Perform GRPO update!
        update()
        episode_in_epoch_cnt = 0

        # Perform self-imitation (unchanged)
        if imitation_buf.cur_size >= 1 and (epoch % args.imitate_freq == 0) and epoch > 0:
            for i_imitate in range(args.imitate_epoch):
                update_imitation()
            logger.store(ImitationBufferSuccess=imitation_buf.success_rate,
                ImitationBufferReturn=imitation_buf.avg_return,
                ImitationBufferAcceptReturn=imitation_buf.baseline,
                ImitationBufferNumTraj=imitation_buf.cur_size)
            # Log info about imitation
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
        # GRPO Change: Removed VVals
        # logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        # GRPO Change: Removed LossV
        # logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        # GRPO Change: Removed DeltaLossV
        # logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        # to avoid destroying too many blocks, remake the environment
        if (epoch % 50 == 0) and epoch>0:
            env.remake_env()
            # save the imitation learning buffer
            pth = os.path.join(save_path, 'buffer_{}.pth'.format(epoch))
            pickle.dump(imitation_buf, open(pth, 'wb'))
