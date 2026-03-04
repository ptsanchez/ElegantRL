import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DynamicBCTradingDataset(Dataset):
    def __init__(self, csv_path, data_dir, max_step=72):
        self.expert_df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.max_step = max_step
        
        # Pre-load memmap references for extreme speed
        self.tickers = self.expert_df['ticker'].unique()
        self.mmap_data = {t: np.load(f"{data_dir}/{t}.npy", mmap_mode='r') for t in self.tickers}
        self.mmap_time = {t: np.load(f"{data_dir}/{t}_time.npy", mmap_mode='r') for t in self.tickers}

    def __len__(self):
        # We can multiply by 12 if we want to train on every single 5-min candle of the 1hr hold,
        # but to keep it simple and powerful, let's train the agent primarily on the ENTRY conditions.
        return len(self.expert_df)

    def __getitem__(self, idx):
        row = self.expert_df.iloc[idx]
        ticker = row['ticker']
        entry_time = pd.to_datetime(row['entry_time']).value # nanoseconds
        
        # 1. Find the exact row in the .npy file
        time_ary = self.mmap_time[ticker]
        data_idx = np.searchsorted(time_ary, entry_time)
        
        # 2. Extract the 19 Technicals (Exclude target logret at the end)
        techs = self.mmap_data[ticker][data_idx, :-1].astype(np.float32)
        
        # 3. Reconstruct Environment Context (At the moment of entry)
        # Assuming the agent is looking for a trade, it is in STATUS_SEARCHING (0)
        status_feat = 0.0 / 3.0  
        
        # For BC, we can randomize 'time_left' slightly so it learns to enter at any time of day
        cur_step = np.random.randint(0, self.max_step - 12) 
        time_left = (self.max_step - cur_step) / self.max_step
        
        # PnL is 0 because we haven't entered yet
        unrealized_pnl = 0.0 
        
        # 4. Concatenate EXACTLY like `get_state()` in SingleStockTradingEnv
        state = np.hstack((techs, status_feat, time_left, unrealized_pnl)).astype(np.float32)
        
        # 5. Get Targets
        action = int(row['action'])
        net_ret = float(row['net_ret'])
        
        return torch.from_numpy(state), action, net_ret