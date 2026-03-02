import os
import torch as th
from elegantrl.agents.AgentPPO import AgentDiscretePPO
from elegantrl.envs.SingleStockTradingEnv import SingleStockTradingEnv
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent

"""
Single Stock Trading Demo (Multi-Processing Version)
Optimized for Memory Mapped (.npy) Data, Sparse Rewards, and Stability.
"""

def demo_production_training_mp(num_workers: int = 4):
    # ------------------------------------------------------------------
    # 1. Environment Configuration
    # ------------------------------------------------------------------
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data') 
    
    # Define arguments for both Config and the Environment
    env_args = {
        # --- Required by ElegantRL Config ---
        'env_name': 'SingleStockTradingEnv-v4',  # Logging folder name
        'if_discrete': True,                     # PPO type (Discrete vs Continuous)
        
        # --- Required by Your Custom Env ---
        'data_path': data_path,          
        'episode_days': 1,               
        'price_column': 'target_reg_5m_logret', 
        'if_day_trade': True,
        'cost_pct': 1e-4,                
        #'gamma': 0.99,
    }
    
    # ------------------------------------------------------------------
    # 2. Warm-up & Dimensionality Check
    # ------------------------------------------------------------------
    print(f"| Checking data in: {data_path}")
    
    # FILTER: Remove keys that SingleStockTradingEnv doesn't accept
    # This prevents the "TypeError: __init__() got unexpected keyword"
    init_args = {k: v for k, v in env_args.items() 
                 if k not in ['env_name', 'if_discrete', 'state_dim', 'action_dim', 'max_step']}
    
    # Initialize one temporary instance to get dimensions automatically
    try:
        temp_env = SingleStockTradingEnv(**init_args)
    except Exception as e:
        print(f"| CRITICAL: Failed to load environment. Check data path.\nError: {e}")
        return
    
    # Pass these dimensions back to the config so the network builds correctly
    env_args['state_dim'] = temp_env.state_dim
    env_args['action_dim'] = temp_env.action_dim
    env_args['max_step'] = temp_env.max_step 
    
    print(f"| State Dim: {env_args['state_dim']}")
    print(f"| Action Dim: {env_args['action_dim']}")
    print(f"| Max Step: {env_args['max_step']}")

    # ------------------------------------------------------------------
    # 3. Agent Configuration (Config)
    # ------------------------------------------------------------------
    args = Config(agent_class=AgentDiscretePPO, 
                  env_class=SingleStockTradingEnv, 
                  env_args=env_args)
    
    # Scales raw PnL (0.01) to Optimizer Reward (1.0).
    # Essential for Sparse Rewards to be learned efficiently.
    args.reward_scale = 100.0    
    
    # --- Multiprocessing ---
    args.num_workers = num_workers
    
    # --- Production Hyperparameters ---
    # Total interaction steps
    args.break_step = int(1e6)       
    
    # PPO Collection Window (Batch size per worker cycle)
    args.horizon_len = 2048          
    
    # Mini-batch size for GPU updates (Larger is better for noisy financial data)
    args.batch_size = 1024           
    
    # Network Architecture (Robust enough for 112 features)
    args.net_dims = [256, 128]       
    
    # Learning Rate
    args.learning_rate = 1e-4        
    
    # Evaluation Settings
    args.eval_per_step = 20000       # Check progress every 10k steps
    args.eval_times = 4              # Average over 4 episodes to reduce noise

    args.gamma = 0.99       # Discount factor. Should be changed to 1.0 for purely episodic/sparse reward. 0.99 if changing to more dense rewards structure.
    
    # ------------------------------------------------------------------
    # 4. Start Training
    # ------------------------------------------------------------------
    print("| Starting Training Pipeline...")
    print(f"| Reward Scale:  {args.reward_scale}")
    print(f"| Target Steps:  {args.break_step}")
    
    train_agent(args)

if __name__ == "__main__":
    # Ensure this matches your CPU core count roughly (e.g., 4, 6, 8)
    demo_production_training_mp(num_workers=4)