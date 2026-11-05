"""Milestone 7F.1: separate six-action Double DQN for the MuJoCo maze.

Observations, rewards and actions only; never reads environment info or truth.
This checkpoint saves network weights, not a resumable replay buffer.
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MazeQNetwork(nn.Module):
    def __init__(self, observation_dim, n_actions=6, hidden=128):
        super().__init__()
        if observation_dim < 1 or n_actions < 2 or hidden < 1:
            raise ValueError("Invalid DQN network dimensions")
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, observation):
        return self.net(observation)


class MazeDQNAgent:
    def __init__(
        self,
        observation_dim,
        n_actions=6,
        lr=3e-4,
        gamma=0.99,
        replay_capacity=50_000,
        target_update_steps=500,
        seed=0,
        device="auto",
    ):
        if not isinstance(observation_dim, int) or observation_dim < 1:
            raise ValueError("Invalid observation dimension")
        if not isinstance(n_actions, int) or n_actions < 2:
            raise ValueError("Invalid action count")
        if not 0 <= gamma <= 1:
            raise ValueError("Invalid discount factor")
        if replay_capacity < 1 or target_update_steps < 1 or lr <= 0:
            raise ValueError("Invalid learning hyperparameters")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is unavailable")

        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.target_update_steps = target_update_steps
        self.device = torch.device(device)
        # Initializes weights deterministically. Environment randomness comes
        # from independent numpy generators supplied by the training script.
        torch.manual_seed(int(seed))
        self.online = MazeQNetwork(observation_dim, n_actions).to(self.device)
        self.target = MazeQNetwork(observation_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = deque(maxlen=replay_capacity)
        self.updates = 0

    @torch.no_grad()
    def act(self, observation, rng, epsilon=0.0):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.observation_dim,) or not np.all(np.isfinite(obs)):
            raise ValueError("Malformed observation")
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        return int(self.online(tensor).argmax(dim=1).item())

    def store(self, observation, action, reward, next_observation,
              terminated, truncated=False):
        obs = np.asarray(observation, dtype=np.float32)
        nxt = np.asarray(next_observation, dtype=np.float32)
        if obs.shape != (self.observation_dim,) or nxt.shape != obs.shape:
            raise ValueError("Replay observation shape mismatch")
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(nxt)):
            raise ValueError("Replay contains non-finite observation")
        if not 0 <= int(action) < self.n_actions or int(action) != action:
            raise ValueError("Replay action out of range")
        if not np.isfinite(reward):
            raise ValueError("Replay reward must be finite")
        # A time-limit truncation is NOT an MDP termination; bootstrap from its
        # returned final observation. The caller still resets the environment.
        self.replay.append((obs.copy(), int(action), float(reward),
                            nxt.copy(), bool(terminated)))

    def learn(self, rng, batch_size=64):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.replay) < batch_size:
            return None

        indices = rng.choice(len(self.replay), size=batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        obs = torch.as_tensor(np.stack([x[0] for x in batch]), device=self.device)
        actions = torch.as_tensor([x[1] for x in batch],
                                  dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([x[2] for x in batch],
                                  dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(np.stack([x[3] for x in batch]), device=self.device)
        terminated = torch.as_tensor([x[4] for x in batch],
                                     dtype=torch.float32, device=self.device)

        predictions = self.online(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN: online network selects, target network evaluates.
            next_actions = self.online(nxt).argmax(dim=1, keepdim=True)
            next_values = self.target(nxt).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - terminated) * next_values

        loss = nn.functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "updates": self.updates,
            "online_weights": self.online.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, device="auto"):
        # Loading requires recent PyTorch; weights_only avoids generic unpickling.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(
            observation_dim=int(checkpoint["observation_dim"]),
            n_actions=int(checkpoint["n_actions"]),
            gamma=float(checkpoint["gamma"]),
            device=device,
        )
        agent.online.load_state_dict(checkpoint["online_weights"])
        agent.target.load_state_dict(checkpoint["online_weights"])
        agent.online.eval()
        agent.target.eval()
        agent.updates = int(checkpoint["updates"])
        return agent
