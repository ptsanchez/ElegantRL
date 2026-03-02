import torch as th
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent
from elegantrl.envs.SingleStockTradingEnv import SingleStockTradingEnv

# --- CONFIGURATION ---
BC_MODEL_PATH = "ppo_actor_bc.pth"
DATA_PATH = "./data"

def finetune_ppo():
    # 1. Setup Config
    args = Config(agent_class=AgentPPO, env_class=SingleStockTradingEnv)
    args.env_args = {
        'data_path': DATA_PATH,
        'if_day_trade': True,
        'episode_days': 1
    }
    
    # Environment dimensions (must match BC training)
    # Usually we get these from a dummy env
    env = SingleStockTradingEnv(**args.env_args)
    args.state_dim = env.state_dim
    args.action_dim = 3 # Flat, Long, Short
    args.if_discrete = True
    
    args.net_dims = [256, 256]
    args.batch_size = 512
    args.target_step = args.max_step * 4
    args.repeat_times = 8
    args.learning_rate = 2e-5 # Lower learning rate for fine-tuning
    
    # 2. Initialize Agent
    agent = AgentPPO(args.net_dims, args.state_dim, args.action_dim, gpu_id=0, args=args)
    
    # 3. Load BC Weights
    print(f"Loading BC-trained Actor weights from {BC_MODEL_PATH}...")
    try:
        agent.act.load_state_dict(th.load(BC_MODEL_PATH))
        print("Successfully loaded Actor weights.")
    except FileNotFoundError:
        print(f"Warning: {BC_MODEL_PATH} not found. Starting from scratch.")

    # 4. Optional: Warm-start Critic
    # You could run a few collection steps and update ONLY the critic here.
    # For simplicity, we proceed to full PPO.

    # 5. Start Training
    print("Starting PPO Fine-tuning...")
    # train_agent(args) # This would normally run the full pipeline
    
    # Note: In a real scenario, you would pass the pre-initialized agent 
    # to the training loop, or modify the trainer to load weights.
    # Here we show the conceptual setup.
    
    print("Agent is ready for fine-tuning. Dynamic exits will be learned via RL exploration.")

if __name__ == "__main__":
    finetune_ppo()
