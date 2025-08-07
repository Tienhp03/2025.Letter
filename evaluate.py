import torch
import numpy as np
import math
from ppo import PPO
from env_urllc import ENV_paper

def evaluate_model(ppo_agent, model_path, env, num_episodes=50, max_ep_len=1000):
    """
    Evaluate the PPO model over multiple episodes and print average results.

    Args:
        ppo_agent (PPO): Initialized PPO agent.
        model_path (str): Path to the saved model file.
        env (gym.Env): Environment for evaluation.
        num_episodes (int): Number of episodes to evaluate (default: 50).
        max_ep_len (int): Maximum steps per episode (default: 1000).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppo_agent.policy.load_state_dict(torch.load(model_path))
    ppo_agent.policy.eval()

    total_powers = []
    total_rewards = []
    total_delay_violations = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_power = 0.0
        episode_reward = 0.0
        delay_violation = 0

        for t in range(1, max_ep_len + 1):
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action_mean = ppo_agent.policy.actor(state_tensor)
            action = action_mean.cpu().numpy()
            next_state, reward, done, info = env.step(action)
            state = next_state

            power = info.get('power', 0.0)
            delay_violation = info.get('delay_violation', 0)
            episode_power += power
            episode_reward += reward

            if done:
                break

        total_powers.append(episode_power)
        total_rewards.append(episode_reward)
        total_delay_violations.append(delay_violation)

    avg_total_power = np.mean(total_powers)
    avg_average_power = np.mean([power / max_ep_len for power in total_powers])
    avg_total_reward = np.mean(total_rewards)
    avg_delay_violations = np.mean(total_delay_violations)

    print("\n--- Average Results after 50 Episodes ---")
    print(f"Average total power: {avg_total_power:.2f}")
    print(f"Average power per episode: {avg_average_power:.2f}")
    print(f"Average total reward: {avg_total_reward:.2f}")
    print(f"Average delay violations: {avg_delay_violations:.2f}")

if __name__ == "__main__":
    # Example usage
    lambda_rate = 200
    D_max = 5
    xi = 0.01
    max_power = -(2 ** (lambda_rate/200) - 1) / (math.log10(1 - (xi ** (D_max ** -1) / D_max)))
    snr_feedback = True
    harq_type = 'IR'

    env = ENV_paper(lambda_rate, D_max, xi, max_power, snr_feedback, harq_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim, lr_actor=0.0002, lr_critic=0.0002, gamma=0.99, lambda_gae=0, K_epochs=80, eps_clip=0.2, has_continuous_action_space=True, action_std=0.5, minibatch_size=128)
    model_path = 'ppo_best_model_paper.pth'
    evaluate_model(ppo_agent, model_path, env)