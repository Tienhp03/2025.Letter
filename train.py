import math
import numpy as np
import random
import torch
from env_urllc import ENV_paper
from ppo import PPO

def train_ppo():
    # Thiết lập seed để đảm bảo tái hiện
    seed = 2025 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu sử dụng GPU
    np.random.seed(seed)
    random.seed(seed)

    # Environment hyperparameters
    lambda_rate = 300
    D_max = 5
    xi = 0.01
    max_power = 1 # W - 20 dBm
    snr_feedback = True
    harq_type = 'IR'

    env = ENV_paper(lambda_rate, D_max, xi, max_power, snr_feedback, harq_type)

    # State and action space dimensions
    state_dim = env.observation_space.shape[0]
    has_continuous_action_space = True
    action_dim = env.action_space.shape[0] if has_continuous_action_space else env.action_space.n

    # Training hyperparameters
    max_training_timesteps = int(2000000)
    max_ep_len = 1000
    print_freq = max_ep_len * 4
    log_freq = max_ep_len * 2
    action_std = 0.5
    action_std_decay_rate = 0.005
    min_action_std = 0.1
    action_std_decay_freq = 50000
    update_timestep = 2048
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lambda_gae = 0.95
    lr_actor = 0.0002
    lr_critic = 0.0002

    # Print hyperparameters
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps:", max_training_timesteps)
    print("max timesteps per episode:", max_ep_len)
    print("log frequency:", log_freq, "timesteps")
    print("printing average reward over episodes in last:", print_freq, "timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension:", state_dim)
    print("action space dimension:", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a continuous action space policy")
    print("starting std of action distribution:", action_std)
    print("decay rate of std of action distribution:", action_std_decay_rate)
    print("minimum std of action distribution:", min_action_std)
    print("decay frequency of std of action distribution:", action_std_decay_freq, "timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency:", update_timestep, "timesteps")
    print("PPO K epochs:", K_epochs)
    print("PPO epsilon clip:", eps_clip)
    print("discount factor (gamma):", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor:", lr_actor)
    print("optimizer learning rate critic:", lr_critic)
    print("============================================================================================")

    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, lambda_gae, K_epochs, eps_clip, has_continuous_action_space, action_std, minibatch_size=128)

    # Training loop
    best_reward = -float('inf')
    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            if i_episode == 10000:
                print(info)

            if time_step % update_timestep == 0:
                ppo_agent.update()

            if done:
                break

        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        print(f'Episode {i_episode}: Reward = {current_ep_reward}')

        if current_ep_reward >= best_reward:
            print(f"Saving better model at episode {i_episode} with reward {current_ep_reward}")
            best_reward = current_ep_reward
            # Sửa tên file để bao gồm D_max và lambda_rate
            model_path = f'ppo_best_model_D_max_{D_max}_lambda_{lambda_rate}_xi_{xi}.pth'
            torch.save(ppo_agent.policy.state_dict(), model_path)

        i_episode += 1

    return ppo_agent, env

if __name__ == "__main__":
    train_ppo()