import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.config import Config
import numpy as np

# --- CONFIGURATION ---
DATASET_PATH = "BC_state_action_dataset.pth"
MODEL_SAVE_PATH = "ppo_actor_bc.pth"
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
EPOCHS = 20

def train_bc():
    # 1. Load Dataset
    if not th.cuda.is_available():
        device = th.device("cpu")
    else:
        device = th.device("cuda:0")
        
    print(f"Loading dataset from {DATASET_PATH}...")
    data = th.load(DATASET_PATH)
    states = data['states']
    actions = data['actions']
    
    # Calculate Class Weights
    # Flat (0) might be dominant, so we weight it less
    class_counts = th.bincount(actions)
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum() * 3.0 # Normalize for 3 classes
    print(f"Class counts: {class_counts}")
    print(f"Suggested Weights: {weights}")

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Agent
    state_dim = states.shape[1]
    action_dim = 3 # Flat, Long, Short
    
    args = Config()
    args.if_discrete = True
    args.learning_rate = LEARNING_RATE
    
    # Simple net_dims
    net_dims = [256, 256]
    
    agent = AgentPPO(net_dims, state_dim, action_dim, gpu_id=0, args=args)
    agent.act.to(device)
    
    optimizer = th.optim.Adam(agent.act.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # 3. Training Loop
    print("Starting Behavior Cloning training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            # Forward pass
            # AgentPPO's Actor normally outputs a distribution or action
            # We need the raw logits for CrossEntropy
            # Looking at ActorPPO in AgentPPO.py: 
            # It usually has a get_action or forward method.
            # Let's assume it has a forward method that returns logits for discrete
            logits = agent.act(batch_states)
            
            loss = criterion(logits, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = th.max(logits.data, 1)
            total += batch_actions.size(0)
            correct += (predicted == batch_actions).sum().item()
            
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 4. Save Model
    th.save(agent.act.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ BC-trained Actor saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_bc()
