import argparse
#from spinup_utils.mpi_tools import mpi_fork
import gym
import utils
from DQN import DQN
import os
from spinup_utils.run_utils import setup_logger_kwargs
#from minecraft import MinecraftEnv
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99) # discount
    #parser.add_argument('--target-kl', type=float, default=0.5) # kl upper bound for updating policy
    parser.add_argument('--seed', '-s', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--cpu', type=int, default=1) # number of workers, should be 1, mpi incompatible with python 3.9?
    parser.add_argument('--gpu', default='0') # -1 if use cpu, otherwise select the gpu id
    parser.add_argument('--steps', type=int, default=1000) # sample steps per epoch (buffer size * workers)
    parser.add_argument('--epochs', type=int, default=1000) # epoch number
    parser.add_argument('--save-path', type=str, default='checkpoint') # model save path
    parser.add_argument('--exp-name', type=str, default='dqn') # log name
    #parser.add_argument('--mode', type=str, default='DIRECT') # GUI if use real time render
    parser.add_argument('--task', type=str, default='harvest_milk_with_empty_bucket_and_cow') # task_id
    parser.add_argument('--horizon', type=int, default=200) # task horizon. 500 in the current released code

    parser.add_argument('--clip-config-path', type=str, default='mineclip_official/config.yml')
    parser.add_argument('--clip-model-path', type=str, default='mineclip_official/adjust.pth')
    parser.add_argument('--agent-config-path', type=str, default='mineagent/conf.yaml')

    # reward weights
    parser.add_argument('--reward-success', type=float, default=100.)
    parser.add_argument('--reward-clip', type=float, default=1.)
    parser.add_argument('--clip-reward-mode', type=str, default='direct') # how to compute clip reward
    parser.add_argument('--reward-step', type=float, default=-1.) # per-step penalty
    parser.add_argument('--use-dense', type=int, default=0) # use dense reward
    parser.add_argument('--reward-dense', type=float, default=1.) # dense reward weight

    parser.add_argument('--actor-out-dim', type=int, nargs='+', default=[12,3])
    # actor output dimensions. mineagent official: [3,3,4,25,25,8]; my initial implement: [56,3]
    # mineagent with clipped camera space: [3,3,4,5,3] or [12,3]
    # should modify transform_action() in minecraft.py together with this arg

    '''
    # self-imitation learning
    parser.add_argument('--imitate-buf-size', type=int, default=500) # max num of traj to store
    parser.add_argument('--imitate-batch-size', type=int, default=1000) # batchsize for imitation learning
    parser.add_argument('--imitate-freq', type=int, default=100) # how many ppo epochs to run self-imitation
    parser.add_argument('--imitate-epoch', type=int, default=1) # how many self-imitation epochs
    parser.add_argument('--imitate-success-only', type=int, default=0) # save only success trajs into imitation buffer
    parser.add_argument('--save-raw-rgb', type=int, default=0) # save embeddings or rgb for Bohan's work?
    '''

    args = parser.parse_args()
    #print(args)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = os.path.join(args.save_path, '{}-{}-seed{}'.format(args.exp_name, args.task, args.seed))
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    #mpi_fork(args.cpu)  # run parallel code with mpi
    args.exp_name = args.exp_name + '_' + args.task
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # set gpu device
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    DQN(args ,gamma=args.gamma, save_path=args.save_path,seed=args.seed,steps_per_epoch=args.steps,
        epochs=args.epochs,logger_kwargs=logger_kwargs, device=device,
        clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path,
        agent_config_path=args.agent_config_path, action_dim=np.prod(args.actor_out_dim))