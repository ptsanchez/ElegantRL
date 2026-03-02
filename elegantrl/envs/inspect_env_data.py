import os
import pandas as pd
# Ensure this matches your actual file structure
from SingleStockTradingEnv import SingleStockTradingEnv

def inspect_technicals(data_path='./data'):
    print(f"--- Initializing Environment from {data_path} ---")
    
    try:
        # Initialize the environment just like in the training script
        env = SingleStockTradingEnv(
            data_path=data_path,
            if_day_trade=True
        )
    except Exception as e:
        print(f"Failed to initialize environment. Error: {e}")
        return

    # Reset the environment to load a random ticker and time slice
    print("\n--- Resetting Environment to load episode ---")
    state, info = env.reset()
    
    # Extract the necessary data
    ticker = env.current_ticker
    feature_names = env.feature_names
    # After reset, cur_step is 0. We pull the raw technicals for this step.
    tech_values = env.episode_tech_ary[env.cur_step]
    
    print(f"\nLoaded Ticker: {ticker}")
    print(f"Total Technical Features: {len(feature_names)}")
    print(f"Shape of episode_tech_ary: {env.episode_tech_ary.shape}")
    print("-" * 40)
    
    # Verify dimensions match
    if len(feature_names) != len(tech_values):
        print(f"WARNING: Dimension mismatch! Metadata names: {len(feature_names)}, Tech values: {len(tech_values)}")
    
    # Create a Pandas DataFrame for clean tabular viewing
    df_inspection = pd.DataFrame({
        'Feature Name': feature_names,
        'Value (Step 0)': tech_values
    })
    
    # Configure pandas to show all rows so nothing gets truncated in the console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n--- Technical Array Columns & Values ---")
    print(df_inspection)
    
    # Optional: Check for any NaNs or Infs that might have slipped through
    nans = df_inspection['Value (Step 0)'].isna().sum()
    if nans > 0:
        print(f"\n⚠️ WARNING: Found {nans} NaN values in the technical array at step 0!")

if __name__ == '__main__':
    # Make sure your terminal is in the folder containing the './data' directory
    inspect_technicals(data_path='./data')