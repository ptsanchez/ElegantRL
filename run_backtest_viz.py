import os
import torch as th
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from elegantrl.agents.AgentPPO import AgentDiscretePPO
# Note: Adjusted import path based on project structure
from elegantrl.envs.SingleStockTradingEnv import SingleStockTradingEnv

# --- CONFIGURATION ---
MODEL_PATH = "./SingleStockTradingEnv-v2_DiscretePPO_0/actor__000000835584_00000.037.pt"
OHLCV_DIR = "/opt/rws/repos/RWS_LightGBM/data/reprocessed/stocks_m5_mdv_gt_100M"
DATA_PATH = "./data"
NUM_PLOTS = 10
NET_DIMS = [256, 128]
COST_PCT = 1e-4  # 1 basis point transaction cost

# --- HELPERS ---
def load_trained_agent(model_path, state_dim, action_dim, net_dims):
    """Initializes agent with required arguments and loads weights."""
    # Force CPU to avoid device mismatch during visualization
    agent = AgentDiscretePPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=-1)
    
    print(f"| Loading weights from {model_path}")
    # weights_only=False for PyTorch 2.6 compatibility with custom classes
    loaded_obj = th.load(model_path, map_location=th.device('cpu'), weights_only=False)
    
    # Support both state_dict and full model objects
    if isinstance(loaded_obj, dict):
        agent.act.load_state_dict(loaded_obj)
    elif isinstance(loaded_obj, th.nn.Module):
        agent.act.load_state_dict(loaded_obj.state_dict())
    else:
        raise TypeError(f"Unexpected type for loaded model: {type(loaded_obj)}")
    
    agent.act.eval()
    return agent

def get_ohlcv_slice(ticker, timestamps, ohlcv_dir):
    """Joins npy timestamps with raw Parquet OHLCV data."""
    path = os.path.join(ohlcv_dir, f"{ticker}.parquet")
    if not os.path.exists(path):
        print(f"| Warning: OHLCV file for {ticker} not found at {path}")
        return None
    
    full_df = pd.read_parquet(path)
    target_times = pd.to_datetime(timestamps, unit='ns') 
    ohlcv_slice = full_df[full_df['timestamp'].isin(target_times)].copy()
    return ohlcv_slice.sort_values('timestamp')

def plot_episode_with_real_pnl(ohlcv_df, actions, ticker, ep_idx):
    """Generates interactive plots using Parquet prices for PnL calculation."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)

    # 1. Initialize PnL tracking
    real_pnl_history = [0.0] * len(ohlcv_df)
    entry_price = 0.0
    trade_status = 0 # 0: Searching, 1: Long, 2: Short

    # 2. Map actions to OHLCV prices
    for i in range(len(actions)):
        current_close = ohlcv_df['close'].iloc[i]
        
        if trade_status == 0:
            if actions[i] == 1: # OPEN LONG
                entry_price = current_close * (1 + COST_PCT)
                trade_status = 1
            elif actions[i] == 2: # OPEN SHORT
                entry_price = current_close * (1 - COST_PCT)
                trade_status = 2
        
        elif actions[i] != 0 or i == (len(actions) - 1):
            if trade_status == 1: # Close Long
                pnl = (current_close / entry_price) - 1.0
            else: # Close Short
                pnl = 1.0 - (current_close / entry_price)
            
            real_pnl_history[i] = pnl
            trade_status = 0

    # 3. Candlesticks
    fig.add_trace(go.Candlestick(
        x=ohlcv_df['timestamp'], open=ohlcv_df['open'], high=ohlcv_df['high'],
        low=ohlcv_df['low'], close=ohlcv_df['close'], name='Price'
    ), row=1, col=1)

    # 4. Trade Markers
    for i, action in enumerate(actions):
        t = ohlcv_df['timestamp'].iloc[i]
        if action == 1: # LONG ENTRY
            fig.add_annotation(x=t, y=ohlcv_df['low'].iloc[i], text="▲", 
                               font=dict(color="green", size=18), showarrow=False)
        elif action == 2: # SHORT ENTRY
            fig.add_annotation(x=t, y=ohlcv_df['high'].iloc[i], text="▼", 
                               font=dict(color="red", size=18), showarrow=False)
        elif i > 0 and action == 0 and actions[i-1] != 0:
            fig.add_annotation(x=t, y=ohlcv_df['close'].iloc[i], text="✘", 
                               font=dict(color="cyan", size=14), showarrow=False)

    # 5. Cumulative PnL
    fig.add_trace(go.Scatter(x=ohlcv_df['timestamp'], y=np.cumsum(real_pnl_history), 
                             name='Realized PnL', fill='tozeroy', 
                             line=dict(color='#00ffcc')), row=2, col=1)

    fig.update_layout(title=f"Verification Backtest: {ticker} (Episode {ep_idx})", 
                      xaxis_rangeslider_visible=False, template='plotly_dark')
    fig.show()

# --- RUNNER ---
def run_analysis():
    env_args = {
        'data_path': DATA_PATH,
        'episode_days': 1,
        'price_column': 'target_reg_5m_logret',
        'if_day_trade': True,
        'cost_pct': COST_PCT
    }
    env = SingleStockTradingEnv(**env_args)
    agent = load_trained_agent(MODEL_PATH, env.state_dim, env.action_dim, NET_DIMS)
    
    for i in range(NUM_PLOTS):
        state, _ = env.reset()
        ticker = env.current_ticker
        times = env.current_time_slice
        
        actions = []
        done = False
        while not done:
            s_tensor = th.as_tensor(state[np.newaxis], dtype=th.float32)
            a_tensor = agent.act(s_tensor)
            action = a_tensor.detach().cpu().numpy()[0]
            
            state, reward, done, _, _ = env.step(action)
            actions.append(action)
            
        ohlcv_df = get_ohlcv_slice(ticker, times, OHLCV_DIR)
        if ohlcv_df is not None:
            plot_episode_with_real_pnl(ohlcv_df, actions, ticker, i)

run_analysis()