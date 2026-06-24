import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from BC_dataset import DynamicBCTradingDataset

# Import your ElegantRL classes here
from elegantrl.agents.AgentPPO import ActorDiscretePPO, CriticPPO

def run_behavior_cloning():
    # --- CONFIGURATION ---
    CSV_PATH = "Oracle_BC_Expert_Trades_2007_2012_1h_Clean.csv"
    DATA_DIR = "./data/data_mmap"
    BATCH_SIZE = 1024
    EPOCHS = 10
    LR = 3e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dimensions based on your Env
    STATE_DIM = 22  # 19 features + 3 context vars
    ACTION_DIM = 3
    NET_DIMS = [256, 128] # Must match what you will use in RL

    # 1. Initialize Dataset & DataLoader
    dataset = DynamicBCTradingDataset(CSV_PATH, DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Initialize ElegantRL Networks
    print("Initializing Actor and Critic...")
    actor = ActorDiscretePPO(net_dims=NET_DIMS, state_dim=STATE_DIM, action_dim=ACTION_DIM).to(DEVICE)
    critic = CriticPPO(net_dims=NET_DIMS, state_dim=STATE_DIM, action_dim=ACTION_DIM).to(DEVICE)

    # 3. Setup Loss and Optimizers
    # Actor predicts categories (0, 1, 2) -> Cross Entropy
    # Critic predicts continuous returns -> Mean Squared Error
    criterion_actor = nn.CrossEntropyLoss()
    criterion_critic = nn.MSELoss()
    
    opt_actor = Adam(actor.parameters(), lr=LR)
    opt_critic = Adam(critic.parameters(), lr=LR)

    # 4. Training Loop
    print(f"Starting Behavior Cloning on {DEVICE}...")
    for epoch in range(EPOCHS):
        actor.train()
        critic.train()
        
        total_actor_loss = 0
        total_critic_loss = 0
        correct_actions = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for states, actions, returns in pbar:
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            returns = returns.to(DEVICE).unsqueeze(1) # [batch, 1]

            # --- TRAIN ACTOR ---
            # We access `actor.net(actor.state_norm(state))` directly to get the raw logits 
            # needed for CrossEntropyLoss. (Using actor.forward() returns the argmax, which breaks gradients).
            logits = actor.net(actor.state_norm(states))
            loss_a = criterion_actor(logits, actions)
            
            opt_actor.zero_grad()
            loss_a.backward()
            opt_actor.step()
            
            # Track Actor Accuracy
            predictions = logits.argmax(dim=1)
            correct_actions += (predictions == actions).sum().item()

            # --- TRAIN CRITIC ---
            values = critic(states)
            loss_c = criterion_critic(values, returns)
            
            opt_critic.zero_grad()
            loss_c.backward()
            opt_critic.step()

            # --- LOGGING ---
            total_actor_loss += loss_a.item()
            total_critic_loss += loss_c.item()
            total_samples += states.size(0)
            
            pbar.set_postfix({
                'A_Loss': f"{loss_a.item():.4f}", 
                'C_Loss': f"{loss_c.item():.6f}",
                'Acc': f"{(correct_actions/total_samples)*100:.1f}%"
            })

        # Epoch Summary
        avg_a_loss = total_actor_loss / len(dataloader)
        avg_c_loss = total_critic_loss / len(dataloader)
        accuracy = (correct_actions / total_samples) * 100
        print(f"Epoch {epoch+1} Complete | Actor Loss: {avg_a_loss:.4f} | Critic Loss: {avg_c_loss:.6f} | Accuracy: {accuracy:.2f}%")

    # 5. Save the cloned brains
    os.makedirs("./pre_trained_models", exist_ok=True)
    torch.save(actor.state_dict(), "./pre_trained_models/actor_bc.pth")
    torch.save(critic.state_dict(), "./pre_trained_models/critic_bc.pth")
    print("✅ Behavior Cloning Complete. Weights saved!")

if __name__ == "__main__":
    run_behavior_cloning()