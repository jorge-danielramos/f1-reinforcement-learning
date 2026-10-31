# gradient_bandit.py

import numpy as np


ACTION_NAMES = {
    0: "Stay Out",
    1: "Pit Soft",
    2: "Pit Medium",
    3: "Pit Hard",
}


class MaskedGradientBanditAgent:
    """
    Gradient Bandit agent with masked Boltzmann action selection.

    Actions:
        0 = Stay out
        1 = Pit Soft
        2 = Pit Medium
        3 = Pit Hard

    The agent learns preferences H(a) over actions.
    Illegal pit actions are masked so they receive probability 0.
    """

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.01,
        temperature: float = 1.0,
        use_baseline: bool = True,
        seed: int = 42,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.temperature = temperature
        self.use_baseline = use_baseline
        self.rng = np.random.default_rng(seed)

        # Action preferences H(a)
        self.H = np.zeros(n_actions, dtype=np.float32)

        # Running average reward baseline
        self.avg_reward = 0.0
        self.t = 0

    def get_action_mask(self, obs):
        """
        Extract or build legal-action mask from observation.

        Handles two observation formats:
        1. Dict from F1Env with 'action_mask' key: use provided mask
        2. Array observation: extract mask info from indices 4-6

        Mask:
            1 = legal action
            0 = illegal action
        """
        
        # If observation is a dict (from F1Env), extract action_mask directly
        if isinstance(obs, dict) and 'action_mask' in obs:
            return obs['action_mask'].astype(np.float32)
        
        # Fallback for array observations
        if isinstance(obs, dict):
            # If it's a dict but no action_mask, something is wrong
            raise ValueError("Dict observation without 'action_mask' key")
        
        # Legacy array observation format
        used_soft = int(obs[4])
        used_medium = int(obs[5])
        used_hard = int(obs[6])

        mask = np.ones(self.n_actions, dtype=np.float32)

        # Stay out is always legal
        mask[0] = 1.0

        # Cannot pit to already-used compounds
        if used_soft:
            mask[1] = 0.0

        if used_medium:
            mask[2] = 0.0

        if used_hard:
            mask[3] = 0.0

        return mask

    def policy(self, action_mask):
        """
        Masked Boltzmann policy:

        pi(a) = exp(H(a)/T) / sum_b exp(H(b)/T)

        Illegal actions receive probability 0.
        """

        logits = self.H / self.temperature

        # Very negative value removes illegal actions from softmax
        masked_logits = np.where(action_mask == 1, logits, -1e9)

        # Numerically stable softmax
        exp_logits = np.exp(masked_logits - np.max(masked_logits))
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def act(self, obs):
        """
        Select action according to masked Boltzmann policy.
        """

        action_mask = self.get_action_mask(obs)
        probs = self.policy(action_mask)

        action = self.rng.choice(self.n_actions, p=probs)

        return action, probs, action_mask

    def update(self, action, reward, probs, action_mask):
        """
        Gradient Bandit update.

        If A_t is selected:

            H(A_t) <- H(A_t) + alpha (R_t - b_t)(1 - pi(A_t))

        For legal non-selected actions:

            H(a) <- H(a) - alpha (R_t - b_t) pi(a)
        """

        self.t += 1

        # Running average reward baseline
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.t
            advantage = reward - self.avg_reward
        else:
            advantage = reward

        # Update only legal actions
        for a in range(self.n_actions):
            if action_mask[a] == 0:
                continue

            if a == action:
                self.H[a] += self.alpha * advantage * (1.0 - probs[a])
            else:
                self.H[a] -= self.alpha * advantage * probs[a]


def train_gradient_bandit(
    env,
    num_episodes: int = 500,
    alpha: float = 0.01,
    temperature: float = 1.0,
    use_baseline: bool = True,
    seed: int = 42,
):
    """
    Train Gradient Bandit on a Gymnasium environment.

    Handles both standard Gymnasium and F1Env environments:
    - Standard: env.step(action) returns (obs, reward, terminated, truncated, info)
    - F1Env: env.step(action) returns (obs, reward, terminated, info)

    Args:
        env: Gymnasium environment (standard or F1Env)
        num_episodes: Number of training episodes
        alpha: Learning rate
        temperature: Softmax temperature for exploration
        use_baseline: Whether to use reward baseline
        seed: Random seed
    """

    agent = MaskedGradientBanditAgent(
        n_actions=env.action_space.n,
        alpha=alpha,
        temperature=temperature,
        use_baseline=use_baseline,
        seed=seed,
    )

    episode_returns = []
    episode_lengths = []
    action_counts = np.zeros(env.action_space.n, dtype=np.int32)

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, probs, action_mask = agent.act(obs)

            step_result = env.step(action)
            
            # Handle both 4 and 5 return values from env.step()
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, terminated, info = step_result
                done = terminated

            agent.update(action, reward, probs, action_mask)

            total_reward += reward
            steps += 1
            action_counts[action] += 1

            obs = next_obs

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    metrics = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "action_counts": action_counts,
    }

    return agent, metrics


def evaluate_gradient_bandit(
    agent,
    env,
    num_episodes: int = 100,
    seed: int = 1000,
):
    """
    Evaluate a trained Gradient Bandit agent.
    
    Handles both standard Gymnasium and F1Env environments:
    - Standard: env.step(action) returns (obs, reward, terminated, truncated, info)
    - F1Env: env.step(action) returns (obs, reward, terminated, info)
    """

    episode_returns = []
    action_counts = np.zeros(env.action_space.n, dtype=np.int32)
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, probs, action_mask = agent.act(obs)

            step_result = env.step(action)
            
            # Handle both 4 and 5 return values from env.step()
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, terminated, info = step_result
                done = terminated

            total_reward += reward
            steps += 1
            action_counts[action] += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    metrics = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "action_counts": action_counts,
    }

    return metrics