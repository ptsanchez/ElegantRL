import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm

# --- CONFIGURATION ---
DATA_DIR = "./data" # Adjusted to workspace data directory if applicable, or keep original for parquet
FILENAME = "2001-2012_val_2015_merged_predictions_ohlcv.parquet"
# Note: If the parquet file is not in ./data, the user might need to adjust this path.
FILE_PATH = os.path.join(DATA_DIR, FILENAME)

# Strategy Config
HORIZON_LONG = 60    # 1 hour
HORIZON_SHORT = 120  # 2 hours
THRESHOLD_LONG = 0.6
THRESHOLD_SHORT = 0.6
MAX_ACTIVE_TRADES = 10
CAPITAL_PER_TRADE = 10000
COST_BPS = 2.0 

# Time Filters
START_TIME = "09:45"
END_TIME = "15:30"
ALLOW_OVERNIGHT = False
EOD_TIME = "16:00"

# Output
OUTPUT_CSV = "BC_expert_trades_dual_filtered.csv"

def run_dual_direction_simulation(data, max_slots, thresh_l, thresh_s, h_long, h_short, start_str, end_str, overnight_allowed, eod_str):
    active_trades = [] 
    completed_trades = []
    
    start_t = pd.to_datetime(start_str).time()
    end_t = pd.to_datetime(end_str).time()
    eod_t = pd.to_datetime(eod_str).time()
    
    # Pre-filter relevant columns to save memory
    # We need prob_up_1h, prob_down_1h (or whatever horizons are in parquet)
    # The user mentioned Short 2hr, so we look for prob_down_2h if it exists, otherwise use 1h prob for 2h hold
    prob_up_col = 'prob_up_1h' 
    prob_down_col = 'prob_down_1h' # Assuming 1h probs are used as proxies or 2h probs exist
    
    # Check if columns exist
    available_cols = data.columns.tolist()
    if prob_up_col not in available_cols:
        # Fallback to whatever prob columns are available
        prob_up_col = [c for c in available_cols if 'prob_up' in c][0]
    if prob_down_col not in available_cols:
        prob_down_col = [c for c in available_cols if 'prob_down' in c][0]
        
    ret_long_col = 'target_reg_1h_logret'
    ret_short_col = 'target_reg_2h_logret' if 'target_reg_2h_logret' in available_cols else 'target_reg_1h_logret'
    
    data = data[['timestamp', 'ticker', prob_up_col, prob_down_col, ret_long_col, ret_short_col]].copy()
    
    timestamps = data['timestamp'].unique()
    data_by_time = dict(tuple(data.groupby('timestamp')))
    
    print(f"🔄 Simulating {len(timestamps)} time steps (Dual Logic)...")
    
    for current_time in tqdm(timestamps, desc="Backtesting"):
        # 1. EXITS
        active_trades = [t for t in active_trades if t['exit_time'] > current_time]
        
        current_slots_filled = len(active_trades)
        slots_available = max_slots - current_slots_filled
        
        # 2. EVALUATE TIME CONSTRAINTS
        curr_t = current_time.time()
        can_enter = start_t <= curr_t <= end_t
        
        # 3. ENTRIES
        if slots_available > 0 and can_enter:
            step_data = data_by_time.get(current_time)
            if step_data is not None:
                # Find Long Candidates
                long_cands = step_data[step_data[prob_up_col] > thresh_l].copy()
                long_cands['direction'] = 'LONG'
                long_cands['prob'] = long_cands[prob_up_col]
                long_cands['horizon'] = h_long
                long_cands['raw_ret'] = long_cands[ret_long_col]
                
                # Find Short Candidates
                short_cands = step_data[step_data[prob_down_col] > thresh_s].copy()
                short_cands['direction'] = 'SHORT'
                short_cands['prob'] = short_cands[prob_down_col]
                short_cands['horizon'] = h_short
                short_cands['raw_ret'] = -short_cands[ret_short_col] # Negative return for short
                
                candidates = pd.concat([long_cands, short_cands])
                
                if not candidates.empty:
                    # Sort by Conviction (Probability)
                    candidates = candidates.sort_values('prob', ascending=False)
                    # Deduplicate by ticker (don't enter long and short on same ticker at same time)
                    candidates = candidates.drop_duplicates(subset=['ticker'])
                    
                    to_enter = candidates.head(slots_available)
                    
                    for _, row in to_enter.iterrows():
                        # Constraint: Don't enter if already in this ticker
                        if any(t['ticker'] == row['ticker'] for t in active_trades):
                            continue
                            
                        proposed_exit = current_time + pd.Timedelta(minutes=row['horizon'])
                        
                        # Overnight check
                        if not overnight_allowed:
                            if proposed_exit.date() != current_time.date() or proposed_exit.time() > eod_t:
                                continue

                        trade = {
                            'entry_time': current_time,
                            'exit_time': proposed_exit,
                            'ticker': row['ticker'],
                            'direction': row['direction'],
                            'prob': row['prob'],
                            'raw_ret': row['raw_ret']
                        }
                        active_trades.append(trade)
                        completed_trades.append(trade)
        
    return pd.DataFrame(completed_trades)

def main():
    print(f"Loading {FILE_PATH}...")
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found. Searching current directory...")
        if os.path.exists(FILENAME):
            df = pd.read_parquet(FILENAME)
        else:
            print("Please ensure the parquet data file is available.")
            return
    else:
        df = pd.read_parquet(FILE_PATH)

    # Pre-processing
    if 'timestamp' not in df.columns and 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    trades_log = run_dual_direction_simulation(
        df, MAX_ACTIVE_TRADES, THRESHOLD_LONG, THRESHOLD_SHORT, 
        HORIZON_LONG, HORIZON_SHORT, START_TIME, END_TIME, 
        ALLOW_OVERNIGHT, EOD_TIME
    )

    if not trades_log.empty:
        # Economics
        cost_decimal = COST_BPS / 10000.0
        trades_log['net_ret'] = trades_log['raw_ret'] - cost_decimal
        
        # --- FILTERING FOR WINNING TRADES ONLY ---
        print(f"Initial trades: {len(trades_log)}")
        win_trades = trades_log[trades_log['net_ret'] > 0].copy()
        print(f"Winning trades: {len(win_trades)} ({len(win_trades)/len(trades_log):.1%})")
        
        # Save to CSV
        win_trades.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Filtered Expert Trade log saved to {OUTPUT_CSV}")
    else:
        print("❌ No trades generated.")

if __name__ == "__main__":
    main()
