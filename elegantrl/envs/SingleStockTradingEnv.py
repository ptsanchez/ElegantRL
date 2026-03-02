import os
import json
import numpy as np
import numpy.random as rd
import pandas as pd
from typing import Tuple

ARY = np.ndarray

class SingleStockTradingEnv:
    """
    Stock-Agnostic Single Stock Trader
    """
    
    def __init__(self,
                 data_path='./data',  
                 cost_pct=1e-4,       
                 episode_days=1, 
                 price_column='target_reg_5m_logret',
                 if_day_trade=True,
                 **kwargs):
        
        self.data_path = data_path
        self.if_day_trade = if_day_trade
        self.cost_pct = cost_pct
        self.episode_days = episode_days
        self.steps_per_day = 78  # 5 min time resolution
        self.max_step = episode_days * self.steps_per_day
        self.price_column_name = price_column

        # ---------------------------------------------------------
        # 1. Load Metadata & Configure Features
        # ---------------------------------------------------------
        meta_path = os.path.join(self.data_path, 'master_metadata.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}.")
            
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        all_feature_names = self.metadata['feature_names']
        
        # Filter out target columns in addition to time and month columns
        self.feature_names = [
            c for c in all_feature_names 
            if not c.startswith(('target_', 'd1_trend_atr_14', 'spy_d1_atr_14'))
        ]
        self.feature_indices = [all_feature_names.index(c) for c in self.feature_names]
        
        # --- NEW: Define Normalization Parameters ---
        # TSI is bounded [-100, 100] -> dividing by 100 maps to [-1, 1]
        # ADX is bounded [0, 100] -> dividing by 100 maps to [0, 1]
        self.oscillator_scales = {
            'm5_mom_tsi': 100.0,
            'spy_m5_tsi': 100.0,
            'd1_adx_adx': 100.0,
            'd1_adx_plus_di': 100.0,
            'd1_adx_minus_di': 100.0
        }
        
        self.norm_idx = []
        self.norm_scale = []
        for col, scale in self.oscillator_scales.items():
            if col in self.feature_names:
                self.norm_idx.append(self.feature_names.index(col))
                self.norm_scale.append(scale)
                
        self.norm_idx = np.array(self.norm_idx, dtype=int)
        self.norm_scale = np.array(self.norm_scale, dtype=np.float32)
        # --------------------------------------------
                
        # Verify Price Column
        if self.price_column_name not in all_feature_names:
            raise ValueError(f"Price column '{self.price_column_name}' not found.")
        self.price_col_idx = all_feature_names.index(self.price_column_name)
        
        # 2. Discover Tickers
        all_files = os.listdir(self.data_path)
        self.tickers = [f.replace('.npy', '') for f in all_files 
                        if f.endswith('.npy') and not f.endswith('_time.npy')]
        
        if len(self.tickers) == 0:
            raise RuntimeError(f"No .npy data files found in {self.data_path}")

        print(f"| Environment loaded with {len(self.tickers)} tickers.")
        print(f"| Input Features: {len(self.feature_names)} (Targets Removed)")

        # 3. Config
        self.num_techs = len(self.feature_names)
        # Reduced state_dim to +3 since hold_feat is removed
        self.state_dim = self.num_techs + 3
        self.action_dim = 3
        
        # Status Constants
        self.STATUS_SEARCHING = 0
        self.STATUS_HOLDING_LONG = 1
        self.STATUS_HOLDING_SHORT = 2
        self.STATUS_DONE = 3
        
        # Caching
        self.index_cache = {} 
        
        # Runtime placeholders
        self.cur_step = 0
        self.status = self.STATUS_SEARCHING
        self.entry_price = 0.0
        self.entry_step = 0
        self.episode_price_ary = None
        self.episode_tech_ary = None

    def reset(self) -> Tuple[ARY, dict]:
        self.cur_step = 0
        self.status = self.STATUS_SEARCHING
        self.entry_price = 0.0
        self.entry_step = 0
        
        while True:
            ticker = rd.choice(self.tickers)
            
            # 1. Load Data (Mmap)
            try:
                feat_path = os.path.join(self.data_path, f"{ticker}.npy")
                time_path = os.path.join(self.data_path, f"{ticker}_time.npy")
                full_tech_mmap = np.load(feat_path, mmap_mode='r')
                full_time_mmap = np.load(time_path, mmap_mode='r')
            except (FileNotFoundError, ValueError):
                continue 
            
            total_rows = full_tech_mmap.shape[0]
            if total_rows <= self.max_step + 10:
                continue 

            # 2. Determine Start Index
            start_idx = 0
            if self.if_day_trade:
                if ticker not in self.index_cache:
                    timestamps = pd.to_datetime(full_time_mmap, unit='ns') 
                    minutes = timestamps.hour * 60 + timestamps.minute
                    valid_starts = np.where(minutes == 570)[0]
                    valid_starts = valid_starts[valid_starts < (total_rows - self.max_step)]
                    self.index_cache[ticker] = valid_starts
                
                valid_options = self.index_cache[ticker]
                if len(valid_options) == 0: continue
                start_idx = rd.choice(valid_options)
            else:
                start_idx = rd.randint(0, total_rows - self.max_step - 1)

            # 3. Create Episode Slices
            end_idx = start_idx + self.max_step + 1
            
            # Copy data to RAM
            raw_data = full_tech_mmap[start_idx : end_idx, self.feature_indices].copy()
            
            # --- NEW: Apply Vectorized Normalization ---
            # Divides only the specified columns by their respective scales
            if len(self.norm_idx) > 0:
                raw_data[:, self.norm_idx] /= self.norm_scale
            # -----------------------------------------
            
            # Sanity check for NaNs before math
            if np.isnan(raw_data).any(): continue
            
            # Using raw data directly from dataset directly
            self.episode_tech_ary = raw_data

            # ---------------------------------------------

            # Extract Price Column (Log Returns)
            raw_log_returns = full_tech_mmap[start_idx : end_idx, self.price_col_idx]
            self.episode_price_ary = np.concatenate([[0.0], np.cumsum(raw_log_returns)])

            self.current_ticker = ticker
            self.current_time_slice = full_time_mmap[start_idx : end_idx].copy()
            
            break 
        
        return self.get_state(), {}

    def get_state(self) -> ARY:
        # 1. Technical Indicators 
        techs = self.episode_tech_ary[self.cur_step]
        
        # 2. Context Features
        status_feat = self.status / 3.0
        time_left = (self.max_step - self.cur_step) / self.max_step
        
        # 3. Unrealized PnL
        current_log_price = self.episode_price_ary[self.cur_step]
        
        if self.status == self.STATUS_HOLDING_LONG:
            unrealized_pnl = np.exp(current_log_price - self.entry_price) - 1.0
        elif self.status == self.STATUS_HOLDING_SHORT:
            unrealized_pnl = 1.0 - np.exp(current_log_price - self.entry_price)
        else:
            unrealized_pnl = 0.0
            
        state = np.hstack((techs, status_feat, time_left, unrealized_pnl)).astype(np.float32)
        return state

    def step(self, action: int) -> Tuple[ARY, float, bool, bool, dict]:
        decision_log_price = self.episode_price_ary[self.cur_step]
        reward = 0.0
        terminal = False

        # --- ABSOLUTE ACTION MAPPING ---
        # 0 = Flat/Close
        # 1 = Long
        # 2 = Short
        
        # 1. Handle Exits / Reversals
        if self.status == self.STATUS_HOLDING_LONG:
            if action == 0 or action == 2:  # Close or Reverse to Short
                reward = self._calc_pnl_value(decision_log_price, is_exit=True)
                self.status = self.STATUS_SEARCHING # Reset to searching momentarily
                
        elif self.status == self.STATUS_HOLDING_SHORT:
            if action == 0 or action == 1:  # Close or Reverse to Long
                reward = self._calc_pnl_value(decision_log_price, is_exit=True)
                self.status = self.STATUS_SEARCHING

        # 2. Handle Entries (Can happen immediately after closing above for reversals)
        if self.status == self.STATUS_SEARCHING:
            if action == 1: # Open LONG
                self.status = self.STATUS_HOLDING_LONG
                self.entry_price = decision_log_price + np.log(1 + self.cost_pct)
                self.entry_step = self.cur_step
            elif action == 2: # Open SHORT
                self.status = self.STATUS_HOLDING_SHORT
                self.entry_price = decision_log_price + np.log(1 - self.cost_pct)
                self.entry_step = self.cur_step

        # --- TIME STEP ---
        self.cur_step += 1
        
        if self.cur_step >= self.max_step:
            terminal = True
            if self.status in [self.STATUS_HOLDING_LONG, self.STATUS_HOLDING_SHORT]:
                final_price = self.episode_price_ary[self.cur_step]
                reward = self._calc_pnl_value(final_price, is_exit=True)
                self.status = self.STATUS_DONE
            elif self.status == self.STATUS_SEARCHING:
                reward = -0.001 # Lowered since rewards are in log returns, so safer to not take trade then take bad trade 
                self.status = self.STATUS_DONE

        return self.get_state(), float(reward), terminal, False, {}

    def _calc_pnl_value(self, price_level, is_exit=False):
        # Calculate exit price by factoring in the cost_pct on the way out
        if self.status == self.STATUS_HOLDING_LONG:
            exit_price = price_level + np.log(1 - self.cost_pct) if is_exit else price_level
            return np.exp(exit_price - self.entry_price) - 1.0
        elif self.status == self.STATUS_HOLDING_SHORT:
            exit_price = price_level + np.log(1 + self.cost_pct) if is_exit else price_level
            return 1.0 - np.exp(exit_price - self.entry_price)
        return 0.0
    
    def render(self):
        pass

# --- Sanity Check Function (Run directly to test) ---
def check_single_stock_trading_env():
    print("Testing SingleStockTradingEnv with Random Actions...")
    try:
        # Ensure ./data exists and has files before running
        env = SingleStockTradingEnv(data_path='./data', if_day_trade=True)
    except Exception as e:
        print(f"Skipping test: {e}")
        return

    state, info = env.reset()
    print(f"State Shape: {state.shape}")
    print(f"Features: {env.num_techs}, Total State Dim: {env.state_dim}")
    
    total_reward = 0
    for i in range(20):
        # Random action
        action = rd.randint(0, 3)
        state, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        
        status_str = ["SEARCH", "LONG", "SHORT", "DONE"][env.status]
        print(f"Step {i+1}: Act {action} | Status {status_str} | Reward {reward:.5f}")
        
        if done:
            print("Episode Finished. Resetting...")
            env.reset()
            
    print("Test Complete.")

if __name__ == '__main__':
    check_single_stock_trading_env()