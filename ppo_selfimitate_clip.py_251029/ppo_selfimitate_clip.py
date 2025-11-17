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
import wandb
#from clip_model import build_model, tokenize_batch
#from torchvision.transforms import Resize 
#from skimage.transform import resize
from mineclip_official import build_pretrain_model, tokenize_batch, torch_normalize
from minecraft import MinecraftEnv, preprocess_obs, transform_action
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import copy  # --- DPO 변경: 참조 정책 복사를 위해 추가 ---
import pickle


# --- DPO 변경: PPOBuffer 클래스 전체 삭제 ---


# self-imitation learning buffer
# DPO에서는 'Chosen' (선호) 궤적과 'Rejected' (비선호) 궤적을 저장하는 데 사용됩니다.
class SelfImitationBuffer:
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
        self.imitate_success_only = imitate_success_only # baseline 계산 시 사용

        #self.rgb_buf = []
        self.i_saved_traj = 0
        self.agent_model = agent_model

    # eval the trajectory performance and decide to store
    def eval_and_store(self, obs, act, ret, success, rgb=None, save_dir=None):
        '''
        store if success or episode return >= baseline
        if the buffer is full, first-in-first-out
        (DPO에서는 baseline 업데이트 용도로만 사용)
        '''
        if self.cur_size > 0:
            self.baseline = np.mean(self.ret_buf) + 2*np.std(self.ret_buf)
            
        is_good_traj = success or ((not self.imitate_success_only) and (ret >= self.baseline))
        
        if is_good_traj:
            # DPO에서는 store()가 별도로 호출되므로 여기서는 baseline 계산만 수행
            # (단, store()가 아닌 eval_and_store()만 호출할 경우 아래 로직 활성화)
            pass
        
    # --- DPO 변경: baseline 계산과 분리된 순수 저장 함수 ---
    def store(self, obs, act, ret, success, rgb=None, save_dir=None):
        """
        DPO를 위해 외부의 결정에 따라 궤적을 강제로 저장합니다.
        (기존 eval_and_store에서 baseline 체크 로직만 제거된 버전)
        """
        assert self.cur_size <= self.max_size
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.ret_buf.append(ret)
        self.success_buf.append(success)
        
        # 통계 업데이트
        self.success_rate = np.mean(self.success_buf)
        self.avg_return = np.mean(self.ret_buf)

        if self.cur_size < self.max_size:
            self.cur_size += 1
        else: # FIFO
            del(self.obs_buf[0])
            del(self.act_buf[0])
            del(self.ret_buf[0])
            del(self.success_buf[0])

        # save the expert trajectory
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            pth = os.path.join(save_dir, 'traj_{}.pth'.format(self.i_saved_traj))
            pickle.dump([obs, act, ret, success, rgb], open(pth, 'wb'))
            self.i_saved_traj += 1
    # --- DPO 변경 끝 ---

    # get all the data for training. 
    # convert the trajectory list [N * [len, dim]] to transition array [N', dim]
    def get(self):
        if self.cur_size == 0:
            # 비어있는 경우 빈 데이터를 반환 (오류 방지)
            return {'act': torch.empty(0, dtype=torch.long), 'obs': Batch() if self.agent_model == 'mineagent' else torch.empty(0)}

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


# 10/9 update:
# maintain the text embedding 
# compute images embedding (with 15 empty frames at begin) for a trajectory 
# compute constrastive intrinsic rewards for all the steps with a moving window
class CLIPReward:
    # (이 클래스는 DPO에서도 선호도 판별을 위한 보상 계산에 필요하므로 수정 없음)
    def __init__(self, clip_model, device, text):
        self.clip_model = clip_model
        self.device = device
        self.text = text

        # load negative prompts
        with open('negative_prompts.txt', 'r') as f:
            self.neg_text = f.read().splitlines()
        
        with torch.no_grad():
            # pre-compute text embedding
            self.text_emb = self.clip_model.text_encoder(tokenize_batch(self.text + self.neg_text).to(self.device))
            assert self.text_emb.shape[0] == 1+len(self.neg_text)

    # compute all the embeddings for a trajectory, concat the 15 empty frames at beginning
    def compute_all_embeddings(self, imgs):
        video_begin = torch_normalize(np.zeros([1, 15, 3, 160, 256])).to(self.device) # pad 15 frames before reset
        video = imgs.to(self.device) # (1, N, 3, 160, 256)
        video = torch.cat((video_begin, video), 1) # (1, 15+N, 3, 160, 256)

        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float)) # (1, 15+N, 512)

    # compute the intrinsic reward for a 16-frames window
    def reward(self, imgs_emb_window, mode='direct'):
        with torch.no_grad():
            v_emb = self.clip_model.temporal_encoder(imgs_emb_window)
            adapted_video, adapted_text = self.clip_model.reward_adapter(v_emb, self.text_emb)
            v_f = adapted_video / adapted_video.norm(dim=1, keepdim=True) # (1, 512)
            v_f = self.clip_model.logit_scale.exp()*v_f
            t_f= adapted_text / adapted_text.norm(dim=1, keepdim=True) # (1, 512)
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
DPO (Direct Preference Optimization) algorithm implementation:
for every epoch, first play the game to collect trajectories.
Store good trajectories in 'imitation_buf' (Chosen) and bad ones in 'rejection_buf' (Rejected).
Then update the actor by maximizing the log-likelihood of Chosen data
and minimizing it for Rejected data, relative to a 'ref_agent'.
'''
def ppo_selfimitate_clip(args, seed=0, device=None, 
        steps_per_epoch=400, epochs=500, gamma=0.99, 
        pi_lr=1e-4, # vf_lr 제거
        train_pi_iters=80, # DPO 학습 반복 횟수로 사용
        lam=0.95, max_ep_len=1000,
        target_kl=0.01, # DPO에서는 모니터링 용도로만 사용
        save_freq=5, logger_kwargs=dict(), save_path='checkpoint', 
        clip_config_path='', clip_model_path='', agent_config_path=''):

    # (주석은 PPO 기준이므로 일부 맞지 않을 수 있음)

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    # Initialize wandb
    wandb.init(
        project="RL-GPT",
        name=logger_kwargs.get('exp_name', 'dpo'),
        config=vars(args) if hasattr(args, '__dict__') else {}
    )

    # Random seed
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
        
        # --- DPO 변경: 가치 함수(critic) 및 관련 feature_net 제거 ---
        # feature_net_v = copy.deepcopy(feature_net) # 삭제
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=args.actor_out_dim,
            device=device,
            **agent_config['actor'],
            activation='tanh',
        )
        # critic = Critic(...) # 삭제
        
        mine_agent = MineAgent(
            actor=actor, 
            critic=None, # Critic 대신 None 전달
            deterministic_eval=False
        ).to(device) 
        mine_agent.eval()
        
    elif args.agent_model == 'cnn':
        # --- DPO 변경: CNNActorCritic도 critic=None을 지원해야 함 ---
        mine_agent = utils.CNNActorCritic(
            action_dim=args.actor_out_dim,
            deterministic_eval=False,
            critic=None # critic=None을 처리하도록 클래스 수정 필요
        ).to(device)
        mine_agent.eval()
    else:
        raise NotImplementedError

    # --- DPO 변경: 참조 정책(Reference Policy) 생성 ---
    ref_agent = copy.deepcopy(mine_agent)
    ref_agent.eval()
    for param in ref_agent.parameters():
        param.requires_grad = False
    logger.log('Reference policy (ref_agent) created and frozen.')
    # --- DPO 변경 끝 ---


    # Count variables
    var_counts = (utils.count_vars(mine_agent), utils.count_vars(model_clip))
    logger.log('\nNumber of parameters: \t agent: %d, \t mineclip: %d\n'%var_counts)

    # --- DPO 변경: PPOBuffer 삭제, 'Rejected' 버퍼 추가 ---
    # buf = PPOBuffer(...) # 삭제
    
    # 'Chosen' (선호) 궤적 버퍼
    imitation_buf = SelfImitationBuffer(agent_act_dim, args.imitate_buf_size, args.imitate_success_only, args.agent_model)
    # 'Rejected' (비선호) 궤적 버퍼
    rejection_buf = SelfImitationBuffer(agent_act_dim, args.imitate_buf_size, args.imitate_success_only, args.agent_model)
    # --- DPO 변경 끝 ---
    

    # --- DPO 변경: PPO 손실(compute_loss_pi, compute_loss_v) 삭제 ---
    # --- DPO 변경: BC 손실(compute_loss_imitation) 삭제 ---

    # --- DPO 변경: DPO 손실 함수 구현 ---
    def compute_loss_dpo(policy_agent, ref_agent, chosen_batch, rejected_batch, beta=0.1):
        """
        DPO 손실을 계산합니다.
        chosen_batch/rejected_batch: {'obs': ..., 'act': ...} 형태의 딕셔너리
        """
        
        # 1. 선호(Chosen) 데이터에 대한 Log-Probabilities
        obs_chosen, act_chosen = chosen_batch['obs'], chosen_batch['act'].to(device)
        if args.agent_model == 'mineagent':
            obs_chosen.to_torch(device=device)
        else:
            obs_chosen = obs_chosen.to(device)

        pi_dist_chosen = policy_agent(obs_chosen).dist
        logp_pi_chosen = pi_dist_chosen.log_prob(act_chosen)
        
        with torch.no_grad():
            ref_dist_chosen = ref_agent(obs_chosen).dist
            logp_ref_chosen = ref_dist_chosen.log_prob(act_chosen)

        # 2. 비선호(Rejected) 데이터에 대한 Log-Probabilities
        obs_rejected, act_rejected = rejected_batch['obs'], rejected_batch['act'].to(device)
        if args.agent_model == 'mineagent':
            obs_rejected.to_torch(device=device)
        else:
            obs_rejected = obs_rejected.to(device)

        pi_dist_rejected = policy_agent(obs_rejected).dist
        logp_pi_rejected = pi_dist_rejected.log_prob(act_rejected)
        
        with torch.no_grad():
            ref_dist_rejected = ref_agent(obs_rejected).dist
            logp_ref_rejected = ref_dist_rejected.log_prob(act_rejected)

        # 3. DPO 손실 계산 (r_hat = logp_pi - logp_ref)
        r_hat_chosen = logp_pi_chosen - logp_ref_chosen
        r_hat_rejected = logp_pi_rejected - logp_ref_rejected

        logits_chosen = beta * r_hat_chosen
        logits_rejected = beta * r_hat_rejected
        
        loss_chosen = -F.logsigmoid(logits_chosen).mean()
        loss_rejected = -F.logsigmoid(-logits_rejected).mean() # log(1 - sigmoid(x)) = logsigmoid(-x)

        loss_dpo = (loss_chosen + loss_rejected) / 2
        
        # 부가 정보 (KL은 정책과 참조 정책 간의 차이를 모니터링)
        kl_chosen = (logp_pi_chosen - logp_ref_chosen).mean().item()
        kl_rejected = (logp_pi_rejected - logp_ref_rejected).mean().item()

        return loss_dpo, dict(kl_chosen=kl_chosen, kl_rejected=kl_rejected)
    # --- DPO 변경 끝 ---


    # Set up optimizers
    pi_optimizer = torch.optim.Adam(mine_agent.actor.parameters(), lr=pi_lr)
    # vf_optimizer = torch.optim.Adam(mine_agent.critic.parameters(), lr=vf_lr) # 삭제


    # --- DPO 변경: PPO update() 함수 및 BC update_imitation() 함수 삭제 ---

    # --- DPO 변경: DPO 업데이트 함수 구현 ---
    def update_dpo(dpo_batch_size=128, dpo_train_iters=80):
        mine_agent.train()
        
        # 최소 배치 크기 설정 (초기 학습을 위해 최소 1개만 있어도 학습 가능)
        min_batch_size = min(dpo_batch_size, max(1, min(imitation_buf.cur_size, rejection_buf.cur_size)))
        
        # 학습할 데이터가 충분한지 확인 (최소 1개씩은 있어야 함)
        if imitation_buf.cur_size < 1 or rejection_buf.cur_size < 1:
            logger.log(f'Not enough data for DPO update, skipping. (Chosen: {imitation_buf.cur_size}, Rejected: {rejection_buf.cur_size})')
            # 로깅을 위해 기본값 저장
            logger.store(LossDPO=0.0, KL_Chosen=0.0, KL_Rejected=0.0)
            return
        
        # 실제 사용할 배치 크기 결정 (가능한 만큼 사용)
        actual_batch_size = min(min_batch_size, imitation_buf.cur_size, rejection_buf.cur_size)
        if actual_batch_size < dpo_batch_size:
            logger.log(f'Using smaller batch size: {actual_batch_size} (requested: {dpo_batch_size}, Chosen: {imitation_buf.cur_size}, Rejected: {rejection_buf.cur_size})')

        # --- 개선: reward 차이가 큰 상위 80개 조합 선택 ---
        import itertools
        
        # 궤적 단위로 모든 조합 생성: (chosen_traj_idx, rejected_traj_idx) 쌍
        n_chosen_traj = imitation_buf.cur_size
        n_rejected_traj = rejection_buf.cur_size
        
        all_traj_pairs = list(itertools.product(range(n_chosen_traj), range(n_rejected_traj)))
        total_traj_pairs = len(all_traj_pairs)
        logger.log(f'Total trajectory pairs: {total_traj_pairs} (Chosen traj: {n_chosen_traj}, Rejected traj: {n_rejected_traj})')
        
        # 각 궤적 pair의 reward 차이 계산
        reward_diffs = []
        for chosen_idx, rejected_idx in all_traj_pairs:
            chosen_ret = imitation_buf.ret_buf[chosen_idx]
            rejected_ret = rejection_buf.ret_buf[rejected_idx]
            reward_diff = chosen_ret - rejected_ret  # 차이가 클수록 좋은 pair
            reward_diffs.append((reward_diff, chosen_idx, rejected_idx))
        
        # Reward 차이가 큰 순서로 정렬 (내림차순)
        reward_diffs.sort(reverse=True, key=lambda x: x[0])
        
        # 상위 80개 선택 (또는 가능한 만큼)
        num_selected = min(dpo_train_iters, total_traj_pairs)
        selected_pairs = reward_diffs[:num_selected]
        
        if num_selected > 0:
            max_diff = reward_diffs[0][0]
            min_diff = reward_diffs[num_selected-1][0]
            logger.log(f'Selected {num_selected} pairs with largest reward differences (max diff: {max_diff:.2f}, min diff: {min_diff:.2f})')
        else:
            logger.log('No pairs selected.')
        
        # 선택된 궤적 pair들의 모든 transition을 수집
        chosen_traj_indices = [pair[1] for pair in selected_pairs]
        rejected_traj_indices = [pair[2] for pair in selected_pairs]
        
        # 선택된 궤적들의 데이터 수집
        selected_chosen_obs = [imitation_buf.obs_buf[i] for i in chosen_traj_indices]
        selected_chosen_act = [imitation_buf.act_buf[i] for i in chosen_traj_indices]
        selected_rejected_obs = [rejection_buf.obs_buf[i] for i in rejected_traj_indices]
        selected_rejected_act = [rejection_buf.act_buf[i] for i in rejected_traj_indices]
        
        # 평탄화하여 transition 단위로 변환
        if args.agent_model == 'mineagent':
            data_chosen = {
                'obs': Batch.cat(selected_chosen_obs),
                'act': torch.as_tensor(np.concatenate(selected_chosen_act), dtype=torch.long)
            }
            data_rejected = {
                'obs': Batch.cat(selected_rejected_obs),
                'act': torch.as_tensor(np.concatenate(selected_rejected_act), dtype=torch.long)
            }
        else:
            data_chosen = {
                'obs': torch.as_tensor(np.concatenate(selected_chosen_obs), dtype=torch.float32),
                'act': torch.as_tensor(np.concatenate(selected_chosen_act), dtype=torch.long)
            }
            data_rejected = {
                'obs': torch.as_tensor(np.concatenate(selected_rejected_obs), dtype=torch.float32),
                'act': torch.as_tensor(np.concatenate(selected_rejected_act), dtype=torch.long)
            }
        
        n_chosen = data_chosen['act'].shape[0]
        n_rejected = data_rejected['act'].shape[0]
        
        # 실제 사용 가능한 배치 크기 재계산
        actual_batch_size = min(actual_batch_size, n_chosen, n_rejected)
        
        if n_chosen < 1 or n_rejected < 1:
             logger.log('Not enough data after selection, skipping.')
             logger.store(LossDPO=0.0, KL_Chosen=0.0, KL_Rejected=0.0)
             return
        
        # 선택된 데이터로 학습 (80번 반복)
        for i in range(dpo_train_iters):
            pi_optimizer.zero_grad()
            
            # 배치 샘플링 (actual_batch_size 사용)
            idxs_chosen = np.random.randint(0, n_chosen, size=actual_batch_size)
            idxs_rejected = np.random.randint(0, n_rejected, size=actual_batch_size)

            try:
                # (MineAgent.Batch 클래스가 인덱싱을 지원해야 함)
                batch_chosen = {'obs': data_chosen['obs'][idxs_chosen], 'act': data_chosen['act'][idxs_chosen]}
                batch_rejected = {'obs': data_rejected['obs'][idxs_rejected], 'act': data_rejected['act'][idxs_rejected]}
            except TypeError:
                logger.log("CRITICAL ERROR: MineAgent.Batch does not support indexing (__getitem__). Update cannot proceed.")
                break
            except Exception as e:
                logger.log(f"Error during DPO batch sampling: {e}")
                break

            loss_dpo, dpo_info = compute_loss_dpo(mine_agent, ref_agent, 
                                                  batch_chosen, batch_rejected, 
                                                  beta=0.1) # beta는 하이퍼파라미터
            
            loss_dpo.backward()
            pi_optimizer.step()
        
        logger.store(LossDPO=loss_dpo.item(), 
                     KL_Chosen=dpo_info['kl_chosen'], 
                     KL_Rejected=dpo_info['kl_rejected'])
        
        mine_agent.eval()
    # --- DPO 변경 끝 ---


    start_time = time.time()
    saved_traj_cnt = 0 

    # initialize the clip reward model
    clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])

    # Set up steps per epoch for rollout
    local_steps_per_epoch = steps_per_epoch

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        # (test_video 함수는 PPO와 무관하므로 그대로 사용 가능)
        '''
        # save a video of test
        def test_video():
            ...
        '''
        
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            pth = os.path.join(save_path, 'model', 'model_{}.pth'.format(epoch))
            torch.save(mine_agent.state_dict(), pth)


        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0
        ep_rewards = [] # 스텝별 보상 (CLIP 계산 전)
        ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int32)).view(1,1,*env.observation_size) # CLIP 계산용
        ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0

        # --- DPO 변경: 궤적 임시 저장 리스트 ---
        ep_obs_list = [] # 에이전트 입력용 (Batch 또는 np.array)
        ep_acts_list = [] # 에이전트 행동
        ep_rgbs_list = [] # 저장용 (args.save_raw_rgb)
        # --- DPO 변경 끝 ---

        episode_in_epoch_cnt = 0

        
        # rollout in the environment
        mine_agent.train() 
        for t in range(local_steps_per_epoch):
            if args.save_raw_rgb:
                # rgb_list.append(...) 대신 ep_rgbs_list에 저장
                ep_rgbs_list.append(np.asarray(o['rgb'], dtype=np.uint8))

            # --- DPO 변경: PPOBuffer 저장을 임시 리스트 저장으로 변경 ---
            if args.agent_model == 'mineagent':
                batch_o = preprocess_obs(o, device)
                batch_o.to_numpy()  # GPU 메모리 절약을 위해 numpy로 변환 (in-place)
                ep_obs_list.append(batch_o)  # Batch 객체 그대로 저장
            else:
                batch_o_np = torch_normalize(np.asarray(o['rgb'], dtype=np.int32)).view(1,*obs_dim)
                batch_o = torch.as_tensor(batch_o_np, dtype=torch.float32)
                ep_obs_list.append(batch_o_np) # CPU에 저장
            # --- DPO 변경 끝 ---

            with torch.no_grad():
                # --- DPO 변경: 가치(v) 무시 ---
                # (MineAgent.forward_actor_critic이 critic=None일 때 val=None 반환 가정)
                if args.agent_model == 'mineagent':
                    batch_o.to_torch(device=device)  # in-place 변환
                    batch_o_input = batch_o  # 변환된 batch_o 사용
                else:
                    batch_o_input = batch_o.to(device)
                    
                batch_act = mine_agent.forward_actor_critic(batch_o_input)
                a, v, logp = batch_act.act, batch_act.val, batch_act.logp
                # v = v[0] # 삭제 (v는 None일 수 있음)
                logp = logp[0]
                # --- DPO 변경 끝 ---
            
            ep_acts_list.append(a[0].cpu().numpy()) # 행동 저장

            a_env = transform_action(a)
            next_o, r, d, _ = env.step(a_env)
            success = r

            # (보상 계산 로직은 선호도 판별을 위해 동일하게 유지)
            r = r * args.reward_success + args.reward_step
            ep_rewards.append(r)
            ep_obs = torch.cat((ep_obs, 
                torch_normalize(np.asarray(next_o['rgb'], dtype=np.int32)).view(1,1,*env.observation_size)), 1)

            if args.use_dense:
                r_dense = next_o['dense_reward']
                r += r_dense * args.reward_dense
                ep_ret_dense += r_dense

            ep_success += success
            if ep_success > 1:
                ep_success = 1
            ep_ret += r
            ep_len += 1

            # --- DPO 변경: buf.store(...) 및 VVals 로깅 삭제 ---
            # buf.store(batch_o, a[0].cpu().numpy(), r, v, logp) # 삭제
            # logger.store(VVals=v.detach().cpu().numpy()) # 삭제
            # --- DPO 변경 끝 ---
            
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                # (CLIP 보상 계산은 선호도 판별을 위해 동일하게 유지)
                clip_reward_model.compute_all_embeddings(ep_obs)
                ep_rewards_clip = clip_reward_model.compute_all_rewards()
                ep_rewards = np.asarray(ep_rewards) + args.reward_clip * ep_rewards_clip
                ep_ret_clip = np.sum(ep_rewards_clip)
                ep_ret = np.sum(ep_rewards) # ep_ret을 재계산 (기존 r 합 + clip r 합)

                # --- DPO 변경: buf.finish_path() 삭제 및 DPO 버퍼에 저장 ---
                # PPO의 finish_path 로직 모두 삭제

                # DPO 버퍼에 저장할 데이터 준비
                if args.agent_model == 'mineagent':
                    traj_obs = Batch.cat(ep_obs_list)
                else:
                    traj_obs = np.stack(ep_obs_list)
                
                traj_acts = np.stack(ep_acts_list)
                traj_rgbs = np.stack(ep_rgbs_list) if args.save_raw_rgb else None

                # '선호'/'비선호' 결정
                # baseline은 '선호' 버퍼의 통계를 사용
                baseline = imitation_buf.baseline 
                if imitation_buf.cur_size == 0: 
                    baseline = 0 # 초기 베이스라인 (임의의 값, 튜닝 필요)
                    
                is_good_traj = (ep_success > 0) or \
                               ((not args.imitate_success_only) and (ep_ret >= baseline))

                # store 메소드를 사용하여 해당 버퍼에 저장
                if is_good_traj:
                    imitation_buf.store(traj_obs, traj_acts, ep_ret, ep_success, 
                                        traj_rgbs, save_path if args.save_expert_data else None)
                else:
                    rejection_buf.store(traj_obs, traj_acts, ep_ret, ep_success, 
                                        traj_rgbs, None)

                # '선호' 버퍼의 baseline을 업데이트하기 위해 eval_and_store 호출
                # (데이터 저장 없이 baseline 계산만 수행)
                imitation_buf.eval_and_store(traj_obs, traj_acts, ep_ret, ep_success, 
                                             None, None) 
                # --- DPO 변경 끝 ---

                logger.store(EpRet=ep_ret, EpLen=ep_len, EpSuccess=ep_success, 
                             EpRetClip=ep_ret_clip, EpRetDense=ep_ret_dense)
                
                # 리셋
                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_rewards = []
                ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int32)).view(1,1,*env.observation_size)
                ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
                
                # 임시 리스트 비우기
                ep_obs_list, ep_acts_list, ep_rgbs_list = [], [], []

        # --- DPO 변경: PPO update() 및 BC update_imitation() 호출 삭제 ---
        # update() # 삭제
        
        logger.log(f"Epoch {epoch} finished. Updating with DPO...")
        update_dpo(dpo_batch_size=args.imitate_batch_size, 
                   dpo_train_iters=train_pi_iters) 
        
        # if epoch % args.imitate_freq == 0 ... # 삭제
        #    update_imitation() # 삭제
        # --- DPO 변경 끝 ---

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpSuccess', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpRetClip', with_min_and_max=True)
        logger.log_tabular('EpRetDense', with_min_and_max=True)
        # --- DPO 변경: PPO/BC 로깅을 DPO 로깅으로 변경 ---
        # logger.log_tabular('VVals', ... # 삭제
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        # logger.log_tabular('LossPi', ... # 삭제
        # logger.log_tabular('LossV', ... # 삭제
        # logger.log_tabular('DeltaLossPi', ... # 삭제
        # logger.log_tabular('DeltaLossV', ... # 삭제
        # logger.log_tabular('Entropy', ... # 삭제
        # logger.log_tabular('KL', ... # 삭제
        # logger.log_tabular('ClipFrac', ... # 삭제
        # logger.log_tabular('StopIter', ... # 삭제
        # logger.log_tabular('LossImitation', ... # 삭제
        
        logger.log_tabular('LossDPO', with_min_and_max=True)
        logger.log_tabular('KL_Chosen', with_min_and_max=False)
        logger.log_tabular('KL_Rejected', with_min_and_max=False)
        # --- DPO 변경 끝 ---
        logger.log_tabular('Time', time.time()-start_time)
        
        # dump_tabular() 전에 log_current_row의 값을 저장 (dump_tabular() 후에는 비워짐)
        wandb_log_dict = {'epoch': epoch}
        for key, value in logger.log_current_row.items():
            # 스칼라 값만 로깅 (리스트나 복잡한 객체는 제외)
            if isinstance(value, (int, float, np.integer, np.floating)):
                wandb_log_dict[key] = float(value)
            elif isinstance(value, np.ndarray) and value.size == 1:
                wandb_log_dict[key] = float(value.item())
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # 통계값이 튜플로 저장된 경우 (mean, std, min, max)
                if isinstance(value[0], (int, float, np.integer, np.floating)):
                    wandb_log_dict[key] = float(value[0])  # 평균값만 사용
        
        logger.dump_tabular()
        
        # Wandb에 로깅
        wandb.log(wandb_log_dict)

if __name__ == '__main__':
    # (main 함수는 변경 없음)
    parser = argparse.ArgumentParser()

    # basic arguments for PPO
    parser.add_argument('--gamma', type=float, default=0.99) # discount
    parser.add_argument('--target-kl', type=float, default=0.5) # kl upper bound for updating policy
    parser.add_argument('--seed', '-s', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--cpu', type=int, default=1) # number of workers, should be 1
    parser.add_argument('--gpu', default='0') # -1 if use cpu, otherwise select the gpu id
    parser.add_argument('--steps', type=int, default=1000) # sample steps per PPO epoch (buffer size * workers)
    parser.add_argument('--epochs', type=int, default=1000) # PPO epoch number
    parser.add_argument('--save-path', type=str, default='checkpoint') # save dir for model&data. Use /sharefs/baaiembodied/xxx on server
    parser.add_argument('--exp-name', type=str, default='dpo') # exp-name을 dpo로 변경
    

    # arguments for tasks
    parser.add_argument('--task', type=str, default='harvest_milk_with_empty_bucket_and_cow') # programmatic task_id, for single task
    parser.add_argument('--horizon', type=int, default=200) # task horizon. It is 500 in the MineCLIP released code
    parser.add_argument('--use-multi-task', type=int, default=0) # run different tasks across workers
    parser.add_argument('--tasks-config-path', type=str, default='multi_env_config/harvest_milk.yaml') # path to load multi-task configs
    

    # CLIP model and agent model config
    parser.add_argument('--clip-config-path', type=str, default='mineclip_official/config.yml')
    parser.add_argument('--clip-model-path', type=str, default='mineclip_official/adjust.pth')
    parser.add_argument('--agent-model', type=str, default='mineagent') # agent architecture: mineagent, cnn
    parser.add_argument('--agent-config-path', type=str, default='mineagent/conf.yaml') # for mineagent
    parser.add_argument('--actor-out-dim', type=int, nargs='+', default=[12,3])
    ''' 
    actor output dimensions. mineagent official: [3,3,4,25,25,8]; my initial implement: [56,3]
    mineagent with clipped camera space: [3,3,4,5,3] or [12,3]
    should modify transform_action() in minecraft.py together with this arg
    '''

    # reward weights (DPO 선호도 판별에 사용됨)
    parser.add_argument('--reward-success', type=float, default=100.)
    parser.add_argument('--reward-clip', type=float, default=1.)
    parser.add_argument('--clip-reward-mode', type=str, default='direct')  # how to compute clip reward
    parser.add_argument('--reward-step', type=float, default=-1.)  # per-step penalty
    parser.add_argument('--use-dense', type=int, default=0)  # use dense reward
    parser.add_argument('--reward-dense', type=float, default=1.)  # dense reward weight
# ...
# ...
    # self-imitation (DPO) buffer config
    parser.add_argument('--imitate-buf-size', type=int, default=500)  # max num of traj to store (Chosen/Rejected 각각)
    parser.add_argument('--imitate-batch-size', type=int, default=1000)  # batchsize for DPO learning
    parser.add_argument('--imitate-freq', type=int, default=100)  # (사용되지 않음)
    parser.add_argument('--imitate-epoch', type=int, default=1)  # (사용되지 않음)
    parser.add_argument('--imitate-success-only', type=int, default=0)  # baseline 계산 시 성공 궤적만 '좋음'으로 볼지 여부

    # arguments for related research works
    parser.add_argument('--save-all-data', type=int, default=0)  # (사용되지 않음)
    parser.add_argument('--save-expert-data', type=int, default=0)  # 'Chosen' 궤적 저장 여부
    parser.add_argument('--save-raw-rgb', type=int, default=1)  # save rgb images when save the above data; save gif for debug
    parser.add_argument('--use-ss-reward', type=int, default=0)  # (사용되지 않음)
    parser.add_argument('--ss-k', type=int, default=10)  # (사용되지 않음)
    parser.add_argument('--ss-model-path', type=str, default=
        'ss_transformer/trained_on_youtube_interval_1_blocksize_10.pth')  # (사용되지 않음)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = os.path.join(args.save_path, '{}-{}-seed{}'.format(args.exp_name, args.task, args.seed))
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    pth = os.path.join(args.save_path, 'gif')
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = os.path.join(args.save_path, 'model')
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = os.path.join(args.save_path, 'experience_buffer')
    if not os.path.exists(pth):
        os.mkdir(pth)

    args.exp_name = args.exp_name + '_' + args.task
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # set gpu device
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    print('Using device:', device)

    from ppo_selfimitate_clip import ppo_selfimitate_clip
    print('Training DPO (ppo_selfimitate_clip file).')
    ppo_selfimitate_clip(args,
        gamma=args.gamma, save_path=args.save_path, target_kl=args.target_kl,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=device,
        clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path)
