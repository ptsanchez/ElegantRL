import os
import time
import torch as th
import numpy as np
import inspect
from typing import List, Tuple

from elegantrl.agents.AgentPPO import AgentDiscretePPO
from elegantrl.envs.SingleStockTradingEnv import SingleStockTradingEnv
from elegantrl.train.config import Config, build_env

"""
Single Stock Trading Demo (Production-Ready Scale)
This script demonstrates the SingleStockTradingEnv-v2 running with a PPO policy.
Configured for real training on your 317 ticker dataset.
"""

def kwargs_filter(function, kwargs: dict) -> dict:
    """Filter kwargs to only include parameters accepted by the function."""
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}

class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times
        self.eval_per_step = eval_per_step

        self.recorder = []
        print(f"| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}")

    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.total_step < self.eval_step + self.eval_per_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = []
        device = next(actor.parameters()).device
        for _ in range(self.eval_times):
            state, info = self.env_eval.reset()
            episode_reward = 0.0
            episode_step = 0
            for episode_step in range(self.env_eval.max_step):
                s_tensor = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
                # For Discrete PPO, actor(state) returns the argmax action
                action = actor(s_tensor).detach().cpu().numpy()[0]
                
                state, reward, done, truncated, _ = self.env_eval.step(action)
                episode_reward += reward
                if done or truncated:
                    break
            rewards_steps_ary.append((episode_reward, episode_step))

        rewards_steps_ary = np.array(rewards_steps_ary)
        avg_r = rewards_steps_ary[:, 0].mean()
        std_r = rewards_steps_ary[:, 0].std()
        avg_s = rewards_steps_ary[:, 1].mean()

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        save_path = f"{self.cwd}/actor_{self.total_step:010}.pth"
        th.save(actor.state_dict(), save_path)
        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  | {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}")

def train_agent(args: Config):
    args.init_before_training()
    
    # Initialize environment with filtered arguments
    env_kwargs = kwargs_filter(args.env_class.__init__, args.env_args.copy())
    env = args.env_class(**env_kwargs)
    eval_env = args.env_class(**env_kwargs)
    
    # Initialize agent
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    state, _ = env.reset()
    agent.last_state = th.as_tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
    
    # Initialize evaluator
    evaluator = Evaluator(eval_env, eval_per_step=args.eval_per_step, eval_times=args.eval_times, cwd=args.cwd)
        
    '''start training'''
    while True:
        # Explore environment
        buffer_items = agent.explore_env(env, args.horizon_len)
        
        # Update network
        logging_tuple = agent.update_net(buffer_items)
        
        # Evaluate and save
        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        
        if evaluator.total_step > args.break_step:
            break
            
    print(f"| Training Finished. Output saved in {args.cwd}")

def demo_production_training():
    # 1. Environment Arguments for your custom data
    env_args = {
        'env_name': 'SingleStockTradingEnv-v2',
        'data_path': '/opt/rws/repos/RWS_LightGBM/data/reprocessed/final_merged/ml_ready_with_targets/',
        'episode_len': 1,        # 1 day = 78 bars
        'if_discrete': True,
        'max_cache_size': 40,    # Respectful of shared RAM, but enough for speed
    }
    
    # Initialize environment briefly to get state_dim
    init_kwargs = kwargs_filter(SingleStockTradingEnv.__init__, env_args.copy())
    temp_env = SingleStockTradingEnv(**init_kwargs)
    
    env_args['state_dim'] = temp_env.state_dim
    env_args['action_dim'] = temp_env.action_dim
    
    # 2. Configuration
    args = Config(agent_class=AgentDiscretePPO, env_class=SingleStockTradingEnv, env_args=env_args)
    
    # Production Hyperparameters
    args.break_step = 50000          # Total steps
    args.horizon_len = 2048          # Standard PPO collection window
    args.batch_size = 256            # Larger batch for GPU efficiency
    args.eval_per_step = 2000        # Evaluate every 10,000 steps
    args.eval_times = 8              # More trials for stable metrics
    args.net_dims = [256, 128]       # Robust network for 80+ features
    args.learning_rate = 1e-4        # Stable learning rate
    args.reward_scale = 1.0          # Rewards are ROI decimals (e.g. 0.05)
    
    # 3. Device selection
    args.gpu_id = 0 if th.cuda.is_available() else -1
    
    # 4. Start Training
    train_agent(args)

if __name__ == "__main__":
    demo_production_training()
