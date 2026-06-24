import os
import torch as th
import shutil
from elegantrl.agents.AgentPPO import AgentDiscretePPO
from elegantrl.envs.SingleStockTradingEnv import SingleStockTradingEnv
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent, get_rewards_and_steps

def run_fast_sanity_check():
    print("--- Starting Fast Sanity Check ---")
    
    # 1. Setup Local Paths
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data')
    
    # Clean up previous logs
    if os.path.exists('./SingleStockTradingEnv-v2_PPO_0'):
        shutil.rmtree('./SingleStockTradingEnv-v2_PPO_0')

    # 2. Environment Args
    # WE ADD 'env_name' AND 'if_discrete' FOR THE CONFIG OBJECT
    # The 'kwargs_filter' in ElegantRL will strip these before passing to your Env class
    env_args = {
        'env_name': 'SingleStockTradingEnv-v2',  # <--- REQUIRED BY CONFIG
        'if_discrete': True,                     # <--- REQUIRED BY CONFIG
        
        'data_path': data_path,
        'ohlcv_path': '/opt/rws/repos/RWS_LightGBM/data/reprocessed/stocks_m5_mdv_gt_100M',
        'episode_days': 1,
        'price_column': 'target_reg_5m_logret',
        'if_day_trade': True,
        'cost_pct': 1e-4,
        'gamma': 0.99,
    }
    
    # 3. Initialize Temp Env for Dimensions
    # We must filter args manually here because we are calling __init__ directly
    # purely to get the state/action dimensions
    print(f"| Loading environment from {data_path}...")
    
    # Filter out keys that SingleStockTradingEnv doesn't accept
    init_args = {k: v for k, v in env_args.items() 
                 if k not in ['env_name', 'if_discrete', 'state_dim', 'action_dim', 'max_step']}
    
    try:
        temp_env = SingleStockTradingEnv(**init_args)
    except Exception as e:
        print(f"| CRITICAL ERROR: Environment failed to load. {e}")
        return

    # Add dynamic properties to env_args so Config can read them
    env_args['state_dim'] = temp_env.state_dim
    env_args['action_dim'] = temp_env.action_dim
    env_args['max_step'] = temp_env.max_step
    
    print(f"| Env Loaded. State: {env_args['state_dim']}, Action: {env_args['action_dim']}")

    # 4. Agent Configuration
    # Now env_args has everything Config needs ('env_name', 'if_discrete', 'state_dim', etc.)
    args = Config(agent_class=AgentDiscretePPO, 
                  env_class=SingleStockTradingEnv, 
                  env_args=env_args)
    
    # --- Multiprocessing ---
    args.num_workers = 2  
    args.learner_gpus = [0] if th.cuda.is_available() else [-1]
    
    # --- "Fast Run" Hyperparameters ---
    args.break_step = 10000   
    args.horizon_len = 512   
    args.batch_size = 128    
    args.eval_per_step = 512
    args.eval_times = 2      
    args.net_dims = [64, 32] 
    args.reward_scale = 100.0
    
    # 5. Run Training
    print("| Launching Training Loop (Target: ~2000 steps)...")
    try:
        train_agent(args)
        print("\n" + "="*40)
        print("SUCCESS: Pipeline completed without errors.")
        print("="*40)

        # 6. Post-Training Visualization (Sanity Check Render)
        print("| Starting Sanity Check Visualization...")
        
        # Load Best Actor
        actor_path = os.path.join(args.cwd, 'actor.pt')
        if not os.path.exists(actor_path):
            print(f"| Warning: Best actor not found at {actor_path}. Using random actor.")
            # Use current agent.act if train_agent didn't save yet or failed
            # But normally it should be there.
        
        # Initialize Env for Rendering
        render_env = SingleStockTradingEnv(**init_args)
        
        # Initialize Agent and Load Weights
        agent = AgentDiscretePPO(args.net_dims, args.state_dim, args.action_dim, gpu_id=0 if th.cuda.is_available() else -1)
        if os.path.exists(actor_path):
            agent.act.load_state_dict(th.load(actor_path, map_location=th.device('cpu')))
        
        actor = agent.act
        actor.eval()

        # Create Render Directory
        render_dir = os.path.join(args.cwd, 'renders_sanity')
        os.makedirs(render_dir, exist_ok=True)

        print(f"| Rendering 3 episodes to {render_dir}...")
        for i in range(3):
            save_path = os.path.join(render_dir, f"sanity_check_{i}.png")
            
            # Wrap render to inject save_path
            original_render = render_env.render
            render_env.render = lambda: original_render(save_path=save_path)
            
            ret, steps = get_rewards_and_steps(render_env, actor, if_render=True)
            render_env.render = original_render
            
            print(f"| Sanity Episode {i}: Return {ret:.4f}, Steps {steps}")

    except Exception as e:
        print("\n" + "="*40)
        print(f"FAILURE: Pipeline crashed.\nError: {e}")
        print("="*40)

if __name__ == "__main__":
    run_fast_sanity_check()