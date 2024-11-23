import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# Implementation of the Twin Delayed Deep Deterministic Policy Gradient algorithm (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, device, max_action=1):
        super(Actor, self).__init__()
        hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        self.device = device

        self.M = M
        self.N = N
        self.K = K
        self.power_t = power_t

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.max_action = max_action

    def compute_power(self, a):
        G_real = a[:, :self.M ** 2].cpu().data.numpy()
        G_imag = a[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()

        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)
        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))

        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)

        return current_power_t

    def compute_phase(self, a):
        Phi_real = a[:, -2 * self.N:-self.N].detach()
        Phi_imag = a[:, -self.N:].detach()

        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)

    def forward(self, state):
        a = torch.tanh(self.l1(state.float()))
        a = self.bn1(a)
        a = torch.tanh(self.l2(a))
        a = self.bn2(a)
        a = torch.tanh(self.l3(a))

        current_power_t = self.compute_power(a.detach()).expand(-1, 2 * self.M ** 2) / np.sqrt(self.power_t)
        real_normal, imag_normal = self.compute_phase(a.detach())
        real_normal = real_normal.expand(-1, self.N)
        imag_normal = imag_normal.expand(-1, self.N)

        division_term = torch.cat([current_power_t, real_normal, imag_normal], dim=1)

        return self.max_action * a / division_term


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, state, action):
        q = torch.tanh(self.l1(state.float()))
        q = self.bn1(q)
        q = torch.tanh(self.l2(torch.cat([q, action], 1)))
        q = self.l3(q)
        return q


class TD3(object):
    def __init__(self, state_dim, action_dim, M, N, K, power_t, max_action, actor_lr, critic_lr, actor_decay, critic_decay, device, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.device = device
        powert_t_W = 10 ** (power_t / 10)

        # Initialize actor network and target
        self.actor = Actor(state_dim, action_dim, M, N, K, powert_t_W, max_action=max_action, device=device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay)

        # Initialize critic networks and target
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr, weight_decay=critic_decay)

        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr, weight_decay=critic_decay)

        # Hyperparameters
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten().reshape(1, -1)
        return action

    def update_parameters(self, replay_buffer, batch_size=16):
        self.actor.train()
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Target policy smoothing
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.actor.max_action, self.actor.max_action)

        # Compute target Q-value
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = reward + (not_done * self.discount * torch.min(target_Q1, target_Q2)).detach()

        # Update critics
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # Delayed policy update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

    def save(self, file_name):
        torch.save(self.critic_1.state_dict(), file_name + "_critic1")
        torch.save(self.critic_optimizer_1.state_dict(), file_name + "_critic_optimizer1")
        torch.save(self.critic_2.state_dict(), file_name + "_critic2")
        torch.save(self.critic_optimizer_2.state_dict(), file_name + "_critic_optimizer2")
        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

    def load(self, file_name):
        self.critic_1.load_state_dict(torch.load(file_name + "_critic1"))
        self.critic_optimizer_1.load_state_dict(torch.load(file_name + "_critic_optimizer1"))
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(torch.load(file_name + "_critic2"))
        self.critic_optimizer_2.load_state_dict(torch.load(file_name + "_critic_optimizer2"))
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(torch.load(file_name + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
