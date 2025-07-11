
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import math

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        self.mean = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Parameter(T.zeros(n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        

    def forward(self, state):
        x = self.actor(state)
        mean = self.mean(x)
        #clipped_log_std = T.clamp(self.log_std, min=-1.0, max=1.0)
        #std = nn.functional.softplus(self.log_std) + 1e-3
        log_std_clamped = self.log_std.clamp(min=-4.0, max=1.0)
        std = log_std_clamped.exp()
        dist = Normal(mean, std)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, weights_only=False):
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=device, weights_only=weights_only))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, weights_only=False):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=weights_only))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=3e-4, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, chkpt_dir='tmp/ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims, 5e-5, chkpt_dir=chkpt_dir)
        self.memory = PPOMemory(batch_size)

        self.actor_scheduler = T.optim.lr_scheduler.StepLR(self.actor.optimizer, step_size=50, gamma=0.95)
        self.critic_scheduler = T.optim.lr_scheduler.StepLR(self.critic.optimizer, step_size=50, gamma=0.95)

        self.entropy_coef = 0.05  # initial value
        self.entropy_decay = 0.995  # decay rate
        self.min_entropy_coef = 0.0005

        self.logger = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_div": [],
            "adv_mean": [],
            "adv_std": [],
            "returns_mean": [],
            "value_pred_mean": [],
            "returns-value" : [],
        }

    def remember(self, state, action, probs, vals, reward, done):
        action = np.array(action).flatten()
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, weights_only=False):
        self.actor.load_checkpoint(weights_only=weights_only)
        self.critic.load_checkpoint(weights_only=weights_only)
        print("models loaded")

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        with T.no_grad():
            dist = self.actor(state)
            value = self.critic(state)

        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()


    def choose_action_batch(self, observations):
        state = T.tensor(observations, dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state).squeeze()

        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        actions = action.cpu().detach().numpy()
        log_probs = log_prob.cpu().detach().numpy()
        values = value.cpu().detach().numpy()

        return actions, log_probs, values


    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            # Bootstrap next value for GAE
            last_state = state_arr[-1]
            last_state_tensor = T.tensor(last_state, dtype=T.float).unsqueeze(0).to(self.actor.device)
            next_val = self.critic(last_state_tensor).item() if not dones_arr[-1] else 0.0
            vals_arr = np.append(vals_arr, next_val)

            advantages = []
            gae = 0
            for t in reversed(range(len(reward_arr))):
                delta = reward_arr[t] + self.gamma * vals_arr[t + 1] * (1 - int(dones_arr[t])) - vals_arr[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - int(dones_arr[t])) * gae
                advantages.insert(0, gae)

            advantages = T.tensor(advantages, dtype=T.float).to(self.actor.device)
            values = T.tensor(vals_arr[:-1], dtype=T.float).to(self.actor.device)
            returns = advantages + values

            # Handle NaN or extremely small std
            if T.isnan(advantages).any() or advantages.std().item() < 1e-5:
                print(f"[Warning] Advantage std too small or NaN. Clamping.")
                advantages = T.clamp(advantages, -10, 10)

            self.logger["returns_mean"].append(returns.mean().item())
            self.logger["value_pred_mean"].append(values.mean().item())
            self.logger["returns-value"].append(returns.mean().item() - values.mean().item())
            self.logger["adv_mean"].append(advantages.mean().item())
            self.logger["adv_std"].append(advantages.std().item())

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)
                advantages_batch = advantages[batch]
                values_batch = values[batch]
                returns_batch = returns[batch]

                # ✅ Stability: Clip advantages instead of normalizing
                advantages_batch = T.clamp(advantages_batch, -20.0, 20.0)

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()

                new_probs = dist.log_prob(actions).sum(axis=-1)
                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantages_batch * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages_batch
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # ✅ Value loss with clipping
                value_pred_clipped = values_batch + (critic_value - values_batch).clamp(-0.4, 0.4)
                value_losses = (critic_value - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                critic_loss = T.max(value_losses, value_losses_clipped).mean()

                # ✅ Correct entropy computation (mean, not sum)
                entropy = dist.entropy().mean()

                total_loss = actor_loss + 0.25 * critic_loss - self.entropy_coef * entropy
                kl_div = (old_probs - new_probs).mean().item()

                self.logger["policy_loss"].append(actor_loss.item())
                self.logger["value_loss"].append(critic_loss.item())
                self.logger["entropy"].append(entropy.item())
                self.logger["kl_div"].append(kl_div)

                if abs(kl_div) > 1.0:
                    print(f"[Early Stop] KL divergence too high: {kl_div:.3f}")
                    break

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.entropy_coef = max(self.entropy_coef * self.entropy_decay, self.min_entropy_coef)


    def plot_logs(self):
        num_metrics = len(self.logger)
        cols = 3  # Choose how many plots per row
        rows = math.ceil(num_metrics / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axs = axs.flatten()  # flatten in case of single row

        for i, (key, values) in enumerate(self.logger.items()):
            axs[i].plot(values)
            axs[i].set_title(f"{key} over updates")
            axs[i].set_xlabel("Training Step")
            axs[i].set_ylabel(key)
            axs[i].grid(True)

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.savefig("/home/mrc/log_dirs/ppo_all_metrics.png")
        plt.close()



class MultiAgentTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.buffers = {
            'agent1': PPOMemory(agent.memory.batch_size),
            'agent2': PPOMemory(agent.memory.batch_size),
        }

    def remember(self, agent_id, state, action, probs, vals, reward, done):
        assert agent_id in self.buffers
        action = np.array(action).flatten()
        self.buffers[agent_id].store_memory(state, action, probs, vals, reward, done)

    def learn(self, agent_id):
        assert agent_id in self.buffers
        buffer = self.buffers[agent_id]
        self.agent.memory.states = buffer.states
        self.agent.memory.actions = buffer.actions
        self.agent.memory.probs = buffer.probs
        self.agent.memory.vals = buffer.vals
        self.agent.memory.rewards = buffer.rewards
        self.agent.memory.dones = buffer.dones
        self.agent.learn()
        # Don't clear here so we can continue alternating updates if needed

    def clear_memory(self):
        for buffer in self.buffers.values():
            buffer.clear_memory()
