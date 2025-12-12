import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class PPOTrainer:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Replay buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        self.action_dim = action_dim
        
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
            
        # Sample action from distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def compute_returns(self, rewards, dones, next_values):
        """Compute discounted returns"""
        returns = []
        R = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)
            
        return returns
    
    def update(self):
        """Update policy using PPO"""
        if len(self.states) < 32:  # Minimum batch size
            return
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = self.rewards
        dones = self.dones
        
        # Compute returns and advantages
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze().cpu().numpy()
            
        returns = self.compute_returns(rewards, dones, values)
        returns = torch.FloatTensor(returns).to(self.device)
        
        values = torch.FloatTensor(values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old action probabilities
        with torch.no_grad():
            old_action_probs, _ = self.model(states)
            old_action_probs = old_action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            old_log_probs = torch.log(old_action_probs + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(4):
            # Forward pass
            action_probs, state_values = self.model(states)
            state_values = state_values.squeeze()
            
            # Calculate action log probabilities
            selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            log_probs = torch.log(selected_action_probs + 1e-8)
            
            # Calculate ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(state_values, returns)
            
            # Entropy bonus (encourage exploration)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")