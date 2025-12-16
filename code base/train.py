import argparse
import os
import sys # sys와 os는 keep_active.py 생성에 필요
import torch
import time # keep_active.py 생성에 필요

# from spinup_utils.mpi_tools import mpi_fork # 주석 처리됨
import gym
import utils
from spinup_utils.run_utils import setup_logger_kwargs
# from minecraft import MinecraftEnv # 주석 처리됨

# =========================================================================
# 1. [기능 추가] Kaggle 세션 유지 스크립트 생성 함수 정의
# =========================================================================

def create_keep_active_script():
    """pyautogui 기반의 keep_active.py 파일을 현재 디렉토리에 생성합니다."""
    
    file_name = "keep_active.py"
    
    script_content = """
import pyautogui
import time
import sys
import os

# --- 스크립트 실행 전 주의사항 ---
# 이 스크립트는 로컬 PC에서 실행되어야 합니다.
# 실행 중에는 마우스와 키보드를 제어합니다. 중지를 원하시면 터미널에서 Ctrl+C를 누르세요.

def main():
    # 실행 준비 메시지 (로컬 PC의 터미널에 출력됨)
    print("---------------------------------------------------------")
    print("✅ Kaggle 세션 유지 스크립트가 시작되었습니다.")
    print("   스크립트 종료를 원하시면 터미널 창에 Ctrl+C를 누르세요.")
    print("---------------------------------------------------------")
    time.sleep(3) # 실행 전 3초 대기 시간 부여

    try:
        while True:
            pyautogui.typewrite("A")
            x, y = pyautogui.position()
            
            pyautogui.moveTo(x, y - 3, duration=0.2)
            time.sleep(0.2)
            
            pyautogui.moveTo(x, y + 3, duration=0.2)
            time.sleep(0.2)
            
            pyautogui.moveTo(x, y, duration=0.2)
            time.sleep(0.2)

            time.sleep(1.2)
            
    except KeyboardInterrupt:
        print("\\n---------------------------------------------------------")
        print("🛑 스크립트가 사용자 명령(Ctrl+C)으로 중지되었습니다. 🛑")
        print("---------------------------------------------------------")
    except pyautogui.FailSafeException:
        print("\\n🛑 Fail-safe triggered: 마우스가 화면 구석으로 이동하여 스크립트가 종료되었습니다. 🛑")
        sys.exit(1)
    except Exception as e:
        print(f"\\n🛑 예상치 못한 오류 발생: {e} 🛑")
        sys.exit(1)

if __name__ == "__main__":
    if os.name == 'nt':
        os.system("title Kaggle Keep Active")
    else:
        sys.stdout.write('\\33]0;Kaggle Keep Active\\a')
        sys.stdout.flush()

    main()
"""

    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(script_content)
            
        print("\n\n#########################################################")
        print(f"✅ 세션 유지 스크립트 생성 완료: '{file_name}' 저장됨.")
        print("   -> 훈련 중 세션이 끊기지 않도록, 이 파일을 다운로드하여")
        print("      로컬 PC 터미널에서 실행하십시오!")
        print(f"   -> 실행 명령 (Linux/macOS): python {file_name} &")
        print("#########################################################\n")
            
    except Exception as e:
        print(f"\n❌ 파일 생성 중 오류가 발생했습니다: {e}")


# =========================================================================
# 2. train.py 메인 로직 시작
# =========================================================================

if __name__ == '__main__':
    # ----------------------------------------------------
    # [통합] 세션 유지 스크립트 파일 생성
    create_keep_active_script()
    # ----------------------------------------------------

    parser = argparse.ArgumentParser()

    # basic arguments
    parser.add_argument('--gamma', type=float, default=0.99) # discount
    parser.add_argument('--target-kl', type=float, default=0.5) # kl upper bound for updating policy
    parser.add_argument('--seed', '-s', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--cpu', type=int, default=1) # number of workers, should be 1
    parser.add_argument('--gpu', default='0') # -1 if use cpu, otherwise select the gpu id
    parser.add_argument('--steps', type=int, default=1000) # sample steps per PPO epoch (buffer size * workers)
    parser.add_argument('--epochs', type=int, default=500) # PPO epoch number
    parser.add_argument('--save-path', type=str, default='checkpoint') # save dir for model&data. Use /sharefs/baaiembodied/xxx on server

    # GRPO Change: Default experiment name changed from 'ppo' to 'grpo'
    parser.add_argument('--exp-name', type=str, default='grpo') # experiment log name


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

    # reward weights
    parser.add_argument('--reward-success', type=float, default=100.)
    parser.add_argument('--reward-clip', type=float, default=1.)
    parser.add_argument('--clip-reward-mode', type=str, default='direct') # how to compute clip reward
    parser.add_argument('--reward-step', type=float, default=-1.) # per-step penalty
    parser.add_argument('--use-dense', type=int, default=0) # use dense reward
    parser.add_argument('--reward-dense', type=float, default=1.) # dense reward weight

    # self-imitation learning
    parser.add_argument('--imitate-buf-size', type=int, default=500) # max num of traj to store
    parser.add_argument('--imitate-batch-size', type=int, default=1000) # batchsize for imitation learning
    parser.add_argument('--imitate-freq', type=int, default=100) # how many ppo epochs to run self-imitation
    parser.add_argument('--imitate-epoch', type=int, default=1) # how many self-imitation epochs
    parser.add_argument('--imitate-success-only', type=int, default=0) # save only success trajs into imitation buffer

    # arguments for related research works
    parser.add_argument('--save-all-data', type=int, default=0) # save all the collected experience
    parser.add_argument('--save-expert-data', type=int, default=0) # save experience in self-imitation buffer
    parser.add_argument('--save-raw-rgb', type=int, default=1) # save rgb images when save the above data; save gif for debug
    parser.add_argument('--use-ss-reward', type=int, default=0) # experiment for pretrained SS-transformer
    parser.add_argument('--ss-k', type=int, default=10) # prediction horizon for SS transformer
    parser.add_argument('--ss-model-path', type=str, default=
        'ss_transformer/trained_on_youtube_interval_1_blocksize_10.pth') # pretrained SS model path

    args = parser.parse_args()
    #print(args)

    # --- 기존 train.py 파일/디렉토리 생성 로직 ---
    
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

    # mpi_fork(args.cpu) # run parallel code with mpi
    args.exp_name = args.exp_name + '_' + args.task
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # set gpu device
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    print('Using device:', device)

    # GRPO Change: Import the renamed file and function
    from grpo_selfimitate_clip import grpo_selfimitate_clip
    print('Training grpo_selfimitate_clip.')

    # GRPO Change: Call the renamed function
    grpo_selfimitate_clip(args,
        gamma=args.gamma, save_path=args.save_path, target_kl=args.target_kl,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=device,
        clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path,
        agent_config_path=args.agent_config_path)
