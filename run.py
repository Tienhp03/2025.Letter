import math
import os
import torch
import numpy as np
import random
from env_urllc import ENV_paper
from ppo import PPO
from evaluate import evaluate_model

def main():
    # Thiết lập seed để đảm bảo tái hiện
    seed = 2025 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    # Environment parameters
    lambda_rate = 300
    D_max = 5
    xi = 0.01
    max_power = -(2 ** (lambda_rate/200) - 1) / (math.log10(1 - (xi ** (D_max ** -1) / D_max)))
    snr_feedback = False
    harq_type = 'CC'

    # Initialize environment
    env = ENV_paper(lambda_rate, D_max, xi, max_power, snr_feedback, harq_type)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=0.0002,
        lr_critic=0.0002,
        gamma=0.99,
        lambda_gae=0.95,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std=0.5,
        minibatch_size=128
    )

    # Model path for evaluation
    model_path = f'ppo_best_model_D_max_{D_max}_lambda_{lambda_rate}_xi_{xi}.pth'
    # model_path = f'IR_ppo_best_model_D_max_{D_max}_lambda_{lambda_rate}_xi_{xi}.pth'
    # model_path = f'CC_feedback_ppo_best_model_D_max_{D_max}_lambda_{lambda_rate}_xi_{xi}.pth'
    # model_path = f'CC_nofeedback_ppo_best_model_D_max_{D_max}_lambda_{lambda_rate}_xi_{xi}.pth'
    # model_path = 'ppo_best_model_paper.pth'

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")

    # Run evaluation
    evaluate_model(ppo_agent, model_path, env)

if __name__ == "__main__":
    main()