import os
import numpy as np
import pandas as pd
import torch as th
from elegantrl.envs.SingleStockTradingEnv import SingleStockTradingEnv

# --- CONFIGURATION ---
EXPERT_CSV = "BC_expert_trades_dual_filtered.csv"
DATA_PATH = "./data"  # Path to the .npy files
OUTPUT_PATH = "BC_state_action_dataset.pth"

def collect_states_from_expert():
    if not os.path.exists(EXPERT_CSV):
        print(f"Error: {EXPERT_CSV} not found. Run generate_bc_expert_data.py first.")
        return

    expert_df = pd.read_csv(EXPERT_CSV)
    expert_df['entry_time'] = pd.to_datetime(expert_df['entry_time'])
    expert_df['exit_time'] = pd.to_datetime(expert_df['exit_time'])

    env = SingleStockTradingEnv(data_path=DATA_PATH, if_day_trade=True)
    
    collected_data = []

    # Map Tickers to their data
    tickers = env.tickers
    
    # Group expert trades by ticker for efficiency
    expert_by_ticker = dict(tuple(expert_df.groupby('ticker')))

    for ticker in tickers:
        if ticker not in expert_by_ticker:
            # Maybe collect some 'Stay Out' data from tickers with no trades?
            # For now, focus on matching expert trades
            continue
            
        print(f"Processing ticker: {ticker}")
        ticker_trades = expert_by_ticker[ticker]
        
        # We need to align timestamps between expert_df and env's time_mmap
        time_path = os.path.join(DATA_PATH, f"{ticker}_time.npy")
        full_time_mmap = np.load(time_path, mmap_mode='r')
        timestamps = pd.to_datetime(full_time_mmap, unit='ns')
        
        # Load technical features
        feat_path = os.path.join(DATA_PATH, f"{ticker}.npy")
        full_tech_mmap = np.load(feat_path, mmap_mode='r')

        # We will iterate through the days/episodes where expert trades occurred
        for _, trade in ticker_trades.iterrows():
            # Find the index in mmap corresponding to entry_time
            # Note: SingleStockTradingEnv works on indices. We need to find the start_idx.
            entry_idx_matches = np.where(timestamps == trade['entry_time'])[0]
            if len(entry_idx_matches) == 0:
                continue
            entry_idx = entry_idx_matches[0]
            
            # Find exit index
            exit_idx_matches = np.where(timestamps == trade['exit_time'])[0]
            if len(exit_idx_matches) == 0:
                # Fallback: estimate based on horizon if exact match fails
                exit_idx = entry_idx + (60 if trade['direction'] == 'LONG' else 120) // 5 # Assuming 5m bars
            else:
                exit_idx = exit_idx_matches[0]

            # Replay this specific trade window in the environment logic
            # To get accurate 'state', we need to initialize the environment slice
            
            # Slice data
            max_step = exit_idx - entry_idx + 10 # Buffer
            if entry_idx + max_step >= full_tech_mmap.shape[0]:
                max_step = full_tech_mmap.shape[0] - entry_idx - 1
            
            # Manually 'inject' this slice into a temporary env-like state
            raw_data = full_tech_mmap[entry_idx : entry_idx + max_step + 1, env.feature_indices].copy()
            if len(env.norm_idx) > 0:
                raw_data[:, env.norm_idx] /= env.norm_scale
            
            episode_tech_ary = raw_data
            raw_log_returns = full_tech_mmap[entry_idx : entry_idx + max_step + 1, env.price_col_idx]
            episode_price_ary = np.concatenate([[0.0], np.cumsum(raw_log_returns)])
            
            # Simulation loop for this trade
            cur_status = 0 # SEARCHING
            entry_price = 0.0
            
            for step in range(exit_idx - entry_idx + 1):
                if step >= len(episode_tech_ary): break
                
                # Get State
                techs = episode_tech_ary[step]
                status_feat = cur_status / 3.0
                time_left = (max_step - step) / max_step
                current_log_price = episode_price_ary[step]
                
                if cur_status == 1: # LONG
                    unrealized_pnl = np.exp(current_log_price - entry_price) - 1.0
                elif cur_status == 2: # SHORT
                    unrealized_pnl = 1.0 - np.exp(current_log_price - entry_price)
                else:
                    unrealized_pnl = 0.0
                
                state = np.hstack((techs, status_feat, time_left, unrealized_pnl)).astype(np.float32)
                
                # Expert Action
                # If at entry_idx: action 1 or 2
                # If between entry and exit: action 1 or 2
                # If at exit_idx: action 0
                if step == 0:
                    expert_action = 1 if trade['direction'] == 'LONG' else 2
                    # Update internal status for next state
                    cur_status = expert_action
                    entry_price = current_log_price # Simplified for BC state collection
                elif step == (exit_idx - entry_idx):
                    expert_action = 0
                    cur_status = 0
                else:
                    expert_action = 1 if trade['direction'] == 'LONG' else 2
                
                collected_data.append((state, expert_action))

    # --- COLLECT NEUTRAL DATA ---
    print("Collecting neutral (Stay Out) data...")
    # Pick some random slices where no expert trades happened
    # For brevity, we can just take a few random tickers and sample steps
    for ticker in tickers[:10]: # Sample from first 10 tickers
        feat_path = os.path.join(DATA_PATH, f"{ticker}.npy")
        full_tech_mmap = np.load(feat_path, mmap_mode='r')
        total_rows = full_tech_mmap.shape[0]
        
        sample_indices = np.random.choice(range(total_rows - 100), size=50)
        for start_idx in sample_indices:
            raw_data = full_tech_mmap[start_idx : start_idx + 1, env.feature_indices].copy()
            if len(env.norm_idx) > 0:
                raw_data[:, env.norm_idx] /= env.norm_scale
            
            state = np.hstack((raw_data[0], 0.0, 1.0, 0.0)).astype(np.float32)
            collected_data.append((state, 0))

    # Save dataset
    states = [d[0] for d in collected_data]
    actions = [d[1] for d in collected_data]
    
    th.save({
        'states': th.tensor(np.array(states), dtype=th.float32),
        'actions': th.tensor(np.array(actions), dtype=th.long)
    }, OUTPUT_PATH)
    
    print(f"✅ Collected {len(collected_data)} state-action pairs.")
    print(f"Dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    collect_states_from_expert()
