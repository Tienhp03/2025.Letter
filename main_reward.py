import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(data, window_size):
    """Hàm tính trung bình trượt"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Đọc dữ liệu từ file
data_IR = pd.read_csv(r'C:\Users\AVSTC\Desktop\2025.Letter-main\IR_reward_D_max_5_lambda_200_xi_0.01.csv')
data_XP = pd.read_csv(r'C:\Users\AVSTC\Desktop\2025.Letter-main\XP_reward_D_max_3_lambda_200_xi_0.01.csv')
data_CC = pd.read_csv(r'C:\Users\AVSTC\Desktop\2025.Letter-main\CC_reward_D_max_3_lambda_300_xi_0.01.csv')
data_ARQ = pd.read_csv(r'C:\Users\AVSTC\Desktop\2025.Letter-main\ARQ_reward_D_max_3_lambda_300_xi_0.01.csv')

window_size = 10

# Tạo figure với 1 plot
fig, ax = plt.subplots(figsize=(15, 7))

# --- Plot cho tất cả các phương pháp trên cùng một hình ---
for data, label, color in [
    (data_IR, 'IR-HARQ', 'red'),
    (data_XP, 'XP-HARQ', 'blue'),
    (data_CC, 'CC-HARQ', 'green'),
    (data_ARQ, 'ARQ', 'yellow'),
]:
    episodes = data['Episode'].to_numpy() 
    rewards = data['Reward'].to_numpy()

    rewards_smooth = moving_average(rewards, window_size)
    episodes_smooth = episodes[:len(rewards_smooth)]
    std_dev = np.array([np.std(rewards[i:i+window_size]) for i in range(len(rewards_smooth))])

    upper_bound = rewards_smooth + std_dev
    lower_bound = rewards_smooth - std_dev

    ax.plot(episodes_smooth, rewards_smooth, label=label, color=color, linewidth=2)
    ax.fill_between(episodes_smooth, lower_bound, upper_bound, color=color, alpha=0.2)

# Cài đặt định dạng
ax.legend(fontsize=24)
ax.grid(True)
ax.tick_params(axis='both', labelsize=20)

# Thêm nhãn trục
ax.set_xlabel('Episode', fontsize=26)
ax.set_ylabel('Average cumulative reward', fontsize=26)

# Căn chỉnh layout để không che nhãn trục y
plt.tight_layout(rect=[0.06, 0, 1, 1])
plt.show()