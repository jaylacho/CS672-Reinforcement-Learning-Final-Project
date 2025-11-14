import minedojo
import sys
#import imageio
import numpy as np
import time
import copy # deepcopy for task specs is generally safer for minedojo

# reset() bug fixed
# use the multi-discrete action space (3,3,4,25,25,8). For the last dim, allow 0,1,3 only
# further tune and clip the action space, modify transform_action(). 22/9/1

from mineagent.batch import Batch
import torch
from mineclip_official import torch_normalize
from mineagent.features.voxel.flattened_voxel_block import VOXEL_BLOCK_NAME_MAP
import traceback # 오류 추적을 위해 추가

MAX_RETRIES = 3 # 환경 재시작 최대 시도 횟수

def preprocess_obs(obs, device):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.
    """
    B = 1

    def cvt_voxels(vox):
        ret = np.zeros(3*3*3, dtype=np.int64)
        for i, v in enumerate(vox.reshape(3*3*3)):
            if v in VOXEL_BLOCK_NAME_MAP:
                ret[i] = VOXEL_BLOCK_NAME_MAP[v]
        return ret

    # I consider the move and functional action only, because the camera space is too large?
    # construct a 3*3*4*3 action embedding
    def cvt_action(act):
        if act[5]<=1:
            return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
        elif act[5]==3:
            return act[0] + 3*act[1] + 9*act[2] + 72
        else:
            raise Exception('Action[5] should be 0,1,3')

    # [수정됨] PyTorch 연산으로 compass 값을 직접 계산
    # 1. yaw와 pitch 값을 float32 텐서로 직접 생성
    angles_deg = torch.tensor(
        [obs["location_stats"]["yaw"], obs["location_stats"]["pitch"]],
        dtype=torch.float32,
        device=device
    )
    # 2. 라디안으로 변환
    angles_rad = torch.deg2rad(angles_deg)
    # 3. cos, sin 값 계산 (벡터화 연산)
    cos_vals = torch.cos(angles_rad) # [cos(yaw), cos(pitch)]
    sin_vals = torch.sin(angles_rad) # [sin(yaw), sin(pitch)]

    # 4. [cos(yaw), sin(yaw), cos(pitch), sin(pitch)] 순서로 결합
    compass_tensor = torch.stack((cos_vals, sin_vals), dim=1).flatten().unsqueeze(0)

    obs_ = {
        "compass": compass_tensor, # [수정됨] 계산된 텐서 사용
        "gps": torch.as_tensor([obs["location_stats"]["pos"]], device=device),
        "voxels": torch.as_tensor(
            [cvt_voxels(obs["voxels"]["block_name"])], dtype=torch.int64, device=device
        ),
        "biome_id": torch.tensor(
            [int(obs["location_stats"]["biome_id"])], dtype=torch.int64, device=device
        ),
        "prev_action": torch.tensor(
            [cvt_action(obs["prev_action"])], dtype=torch.int64, device=device
        ),
        "prompt": torch.as_tensor(obs["rgb_emb"], device=device).view(B, 512),
        # this is actually the image embedding, not prompt embedding (for single task)
    }
    return Batch(obs=obs_)


# [12, 3] action space, 1 choice among walk, jump and camera
# preserve 5 camera actions

def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]

    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 12
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 == 8: # camera pitch 10
        action[3] = 10
    elif act1 == 9: # camera pitch 14
        action[3] = 14
    elif act1 == 10: # camera yaw 10
        action[4] = 10
    elif act1 == 11: # camera yaw 14
        action[4] = 14

    assert act2 < 3
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)


from collections import deque
class MinecraftEnv:

    def __init__(self, task_id, image_size=(160, 256), max_step=500, clip_model=None, device=None, seed=0,
        dense_reward=False, target_name='cow',  biome=None, **kwargs):
        self.observation_size = (3, *image_size)
        self.action_size = 8
        self.dense_reward = dense_reward
        self.biome = biome
        self.max_step = max_step
        self.cur_step = 0
        self.clip_model = clip_model # use mineclip model to precompute embeddings
        self.device = device
        self.seed = seed
        self.task_id = task_id
        self.image_size = image_size
        self.kwargs = kwargs
        self._target_name = target_name # dense reward를 위해 추가
        self._consecutive_distances = deque(maxlen=2) # dense reward를 위해 추가
        self._distance_min = np.inf # dense reward를 위해 추가

        self.base_env = None # 초기화 전 None 설정
        self.remake_env()
        self.task_prompt = self.base_env.task_prompt
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

    def __del__(self):
        if hasattr(self, 'base_env') and self.base_env is not None:
            self.base_env.close()

    def remake_env(self):
        """
        Minecraft 서버 프로세스를 완전히 종료하고 새로운 환경을 생성합니다.
        서버 충돌 복구에 사용됩니다.
        """
        print('--- Environment remake: Closing old env and spawning new one... ---')
        if hasattr(self, 'base_env') and self.base_env is not None:
            try:
                self.base_env.close()
                time.sleep(1) # 프로세스가 완전히 종료될 시간을 줍니다.
            except Exception as e:
                print(f"Warning: Failed to gracefully close old environment: {e}")
            self.base_env = None # 명확히 None 설정

        # 환경 재생성 로직
        try:
            if not self.dense_reward:
                self.base_env = minedojo.make(
                    task_id=self.task_id,
                    image_size=self.image_size,
                    seed=self.seed,
                    specified_biome=self.biome,
                    fast_reset=True,
                    fast_reset_random_teleport_range_low=0,
                    fast_reset_random_teleport_range_high=100,
                    **self.kwargs)
            else:
                self.base_env = minedojo.make(
                    task_id=self.task_id,
                    image_size=self.image_size,
                    seed=self.seed,
                    specified_biome=self.biome,
                    use_lidar=True,
                    lidar_rays=[
                    (np.pi * pitch / 180, np.pi * yaw / 180, 999)
                    for pitch in np.arange(-30, 30, 6)
                    for yaw in np.arange(-60, 60, 10)],
                    fast_reset=True,
                    fast_reset_random_teleport_range_low=0,
                    fast_reset_random_teleport_range_high=100,
                    **self.kwargs)
            self._first_reset = True
            print('--- Environment remake successful. ---')

        except Exception as e:
            # 환경 생성 자체가 실패하면 심각한 오류이므로 예외를 다시 발생시킵니다.
            print(f"FATAL ERROR: Failed to create new environment. Check MineDojo installation/dependencies. Error: {e}")
            raise e


    def reset(self):
        for retry in range(MAX_RETRIES):
            try:
                # 1. MineDojo 환경 내에서 서버 명령을 통한 환경 정리 (fast_reset=True시 실행됨)
                if not self._first_reset:
                    for cmd in self._reset_cmds:
                        self.base_env.unwrapped.execute_cmd(cmd)
                    self.base_env.unwrapped.set_time(6000)
                    self.base_env.unwrapped.set_weather("clear")
                self._first_reset = False
                self.prev_action = self.base_env.action_space.no_op()

                # 2. 핵심 reset 호출
                obs = self.base_env.reset()
                self.cur_step = 0

                # 3. MineCLIP 및 Dense Reward 처리 (성공 시)
                if self.clip_model is not None:
                    with torch.no_grad():
                        img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int64)).view(1,1,*self.observation_size)
                        img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                        obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                        obs['prev_action'] = self.prev_action

                if self.dense_reward:
                    self._consecutive_distances.clear()
                    self._distance_min = np.inf
                    entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
                    if entity_in_sight:
                        distance = self._distance_min = min(distance, self._distance_min)
                        self._consecutive_distances.append(distance)
                    else:
                        self._consecutive_distances.append(0)

                return obs # 성공적으로 reset 완료

            except Exception as e:
                print(f"--- RESET CRASH DETECTED (Retry {retry+1}/{MAX_RETRIES}) ---")
                print(f"Error: {e}")
                traceback.print_exc(file=sys.stdout)

                if retry < MAX_RETRIES - 1:
                    print("Attempting to remake environment and retry reset...")
                    self.remake_env() # 서버가 죽었으므로 환경을 완전히 새로 만듭니다.
                    time.sleep(2) # 서버가 완전히 켜질 때까지 잠시 대기
                else:
                    print(f"Failed to reset environment after {MAX_RETRIES} retries. Raising error.")
                    raise RuntimeError("MineDojo environment failed to recover after multiple crashes.") from e

    def step(self, act):
        for retry in range(MAX_RETRIES):
            try:
                # 1. 핵심 step 호출
                obs, reward, done, info = self.base_env.step(act)
                self.cur_step += 1
                if self.cur_step >= self.max_step:
                    done = True

                # 2. MineCLIP 및 Dense Reward 처리 (성공 시)
                if self.clip_model is not None:
                    with torch.no_grad():
                        img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int64)).view(1,1,*self.observation_size)
                        img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                        obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                        obs['prev_action'] = self.prev_action

                self.prev_action = act # save the previous action for the agent's observation

                if self.dense_reward:
                    entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
                    nav_reward = 0
                    if entity_in_sight:
                        distance = self._distance_min = min(distance, self._distance_min)
                        self._consecutive_distances.append(distance)
                        nav_reward = self._consecutive_distances[0] - self._consecutive_distances[1]
                    nav_reward = max(0, nav_reward)
                    obs['dense_reward'] = nav_reward

                return obs, reward, done, info # 성공적으로 step 완료

            except Exception as e:
                print(f"--- STEP CRASH DETECTED (Retry {retry+1}/{MAX_RETRIES}) ---")
                print(f"Error: {e}")
                traceback.print_exc(file=sys.stdout)

                if retry < MAX_RETRIES - 1:
                    print("Attempting to remake environment and perform a fresh reset...")
                    self.remake_env() # 서버가 죽었으므로 환경을 완전히 새로 만듭니다.
                    time.sleep(2)
                    # 서버가 새로 만들어졌으니, 다음 시도에서는 reset()부터 다시 시작해야 하지만,
                    # 현재 step()의 목표는 현재 스텝을 완료하는 것이므로,
                    # 여기서는 그냥 exception을 다시 발생시켜 훈련 루프가 reset()을 호출하도록 유도하는 것이 더 일반적입니다.
                    # MineDojo에서는 'done' 상태를 명확히 반환하지 못하고 터지므로, 여기서는 그냥 예외를 다시 발생시켜 훈련 루프에서 잡도록 합니다.
                else:
                    print(f"Failed to step environment after {MAX_RETRIES} retries. Raising error.")
                    raise RuntimeError("MineDojo environment failed to recover during step.") from e

    # for dense reward, find the nearest target in sight
    def _find_distance_to_entity_if_in_sight(self, obs):
        assert self.dense_reward is True
        in_sight, min_distance = False, None
        entities, distances = (
            obs["rays"]["entity_name"],
            obs["rays"]["entity_distance"],
        )
        entity_idx = np.where(entities == self._target_name)[0]
        if len(entity_idx) > 0:
            in_sight = True
            min_distance = np.min(distances[entity_idx])
        return in_sight, min_distance



# import gym
'''
Oct 29
env for multi-process
1. the init function receives a single args
2. not contain CLIP model
3. specially: auto reset an env if done, because all the envs are stepped simultaneously
'''
class MinecraftEnvMP(gym.Env):

    # def __init__(self, task_id, image_size=(160, 256), max_step=500, clip_model=None, device=None, seed=0,
    #              dense_reward=False, target_name='cow'):
    def __init__(self, args):
        self.args = args
        self.observation_size = (3, *args.image_size)
        self.action_size = 8
        #self.dense_reward = bool(args.use_dense)
        if 'biome' in args:
            self._env = minedojo.make(task_id=args.task_id, image_size=args.image_size, seed=args.seed_env, specified_biome=args.biome)
        else:
            self._env = minedojo.make(task_id=args.task_id, image_size=args.image_size, seed=args.seed_env)

        self.max_step = args.horizon
        self.cur_step = 0
        self.task_prompt = self._env.task_prompt
        #self.clip_model = None  # use mineclip model to precompute embeddings
        #self.device = args.device
        self.seed_env = args.seed_env
        self.task_id = args.task_id
        self.image_size = args.image_size
        #self.number_of_episodes = 10000
        self._first_reset = True
        self._reset_cmds = ["/kill @e[type=!player]",
                             "/clear", "/kill @e[type=item]"]

        self.number_of_episodes = 10000

    def __del__(self):
        if hasattr(self, '_env'):
            self._env.close()

    # auto reset after remake
    def remake(self):
        '''
        call this to reset all the blocks and trees
        should modify line 479 in minedojo/tasks/__init__.py, deep copy the task spec dict:
             import deepcopy
             task_specs = copy.deepcopy(ALL_TASKS_SPECS[task_id])
        '''
        self._env.close()
        if 'biome' in self.args:
            self._env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed_env, specified_biome=self.args.biome)
        else:
            self._env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed_env)

        self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')
        return self._env.reset()

    def reset(self):
        if not self._first_reset:
            for cmd in self._reset_cmds:
                self._env.unwrapped.execute_cmd(cmd)
            self._env.unwrapped.set_time(6000)
            self._env.unwrapped.set_weather("clear")
        self._first_reset = False
        self.prev_action = self._env.action_space.no_op()

        obs = self._env.reset()
        self.cur_step = 0

        obs['prev_action'] = self.prev_action

        return obs

    def step(self, act):
        #print(act)
        obs, reward, done, info = self._env.step(act['action'])
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True

        obs['prev_action'] = self.prev_action
        self.prev_action = np.asarray(act['action'])  # save the previous action for the agent's observation

        return obs, reward, done, info


if __name__ == '__main__':
    #print(minedojo.ALL_TASKS_SPECS)
    env = MinecraftEnv(
        task_id="harvest_milk_with_empty_bucket_and_cow",
        image_size=(160, 256),
    )
    reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
    obs = env.reset()
    #print(obs.shape, obs.dtype)
    for t in range (100):
        for i in range(12):
            # act = [42,0] #cam - 이 부분은 transform_action에 맞춰 수정이 필요할 수 있습니다.
            # transform_action이 [12, 3] 액션 공간을 사용하므로, [0-11, 0-2] 형태의 액션을 가정합니다.
            act = [1, 0] # 임시로 '앞으로' 액션 설정
            obs, reward, done, info = env.step(act)
            time.sleep(0.2)
        print('reset')
        # env.base_env.execute_cmd(cmd)는 base_env.reset() 전에만 호출되어야 함 (MinecraftEnv의 reset 로직 참고)
        # 이 테스트 블록은 주석 처리되어 있었으므로 원래대로 유지합니다.

    obs = env.reset()
'''

#print(env.base_env.task_prompt, env.base_env.task_guidance)
obs = env.reset()
print(obs['rgb'].shape, obs['rgb'].dtype) # obs.shape 대신 obs['rgb'].shape 사용
for t in range (1000):

    #act = env.base_env.action_space.no_op()

    if t < 50:
        act = [1,0]    # forward
    elif t < 200:
        act = [0,0]     # stall
    elif t < 400:
        act = [1,2] # attack
    elif t<600:
        act = [8,1] # cam pitch 10 and use (원래 [50,1]이었으나, 12-dim 공간에 맞게 8로 수정)
    elif t<800:
        act = [10,1] # cam yaw 10 and use (원래 [40,1]이었으나, 10으로 수정)
    else:
        act = [1,0]

    obs, reward, done, info = env.step(act)
    print(act, reward)

    if done:
        print("Episode done, forcing reset...")
        obs = env.reset()

'''
