import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(data, window_size):
    """Hàm tính trung bình trượt"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Đọc dữ liệu từ file
data_HS_10 = pd.read_csv(r'C:\Users\dohuy\Desktop\UAV_FSO_THz\UAV_FSO_RF_DRL-main\output 1.2 HS haze\speed_10\0\episode_rewards.csv')
data_SS_10 = pd.read_csv(r'C:\Users\dohuy\Desktop\UAV_FSO_THz\UAV_FSO_RF_DRL-main\output 1.2 SS haze\speed_10\4\episode_rewards.csv')
data_HS_15 = pd.read_csv(r'C:\Users\dohuy\Desktop\UAV_FSO_THz\UAV_FSO_RF_DRL-main\output HS 15\speed_15\2\episode_rewards.csv')
data_SS_15 = pd.read_csv(r'C:\Users\dohuy\Desktop\UAV_FSO_THz\UAV_FSO_RF_DRL-main\output SS 15\speed_15\0\episode_rewards.csv')

window_size = 10

# Tạo figure với 2 subplot xếp dọc
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=False)

# --- Plot cho tốc độ 10 m/s ---
for data, label, color in [
    (data_SS_10, 'Soft-switching (Velocity 10 m/s)', 'red'),
    (data_HS_10, 'Hard-switching (Velocity 10 m/s)', 'blue'),
]:
    episodes = data['Episode Number'].to_numpy() * 64
    rewards = data['Episode Reward'].to_numpy()

    rewards_smooth = moving_average(rewards, window_size)
    episodes_smooth = episodes[:len(rewards_smooth)]
    std_dev = np.array([np.std(rewards[i:i+window_size]) for i in range(len(rewards_smooth))])

    upper_bound = rewards_smooth + std_dev
    lower_bound = rewards_smooth - std_dev

    ax1.plot(episodes_smooth, rewards_smooth, label=label, color=color, linewidth=2)
    ax1.fill_between(episodes_smooth, lower_bound, upper_bound, color=color, alpha=0.2)

ax1.set_ylim(-160, -80)
ax1.set_yticks(np.arange(-160, -80 + 1, 20))  # chia mỗi 20 đơn vị
ax1.legend(fontsize=24)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=20)

# --- Plot cho tốc độ 15 m/s ---
for data, label, color in [
    (data_SS_15, 'Soft-switching (Velocity 15 m/s)', 'red'),
    (data_HS_15, 'Hard-switching (Velocity 15 m/s)', 'blue'),
]:
    episodes = data['Episode Number'].to_numpy() * 64
    rewards = data['Episode Reward'].to_numpy()

    rewards_smooth = moving_average(rewards, window_size)
    episodes_smooth = episodes[:len(rewards_smooth)]
    std_dev = np.array([np.std(rewards[i:i+window_size]) for i in range(len(rewards_smooth))])

    upper_bound = rewards_smooth + std_dev
    lower_bound = rewards_smooth - std_dev

    ax2.plot(episodes_smooth, rewards_smooth, label=label, color=color, linewidth=2)
    ax2.fill_between(episodes_smooth, lower_bound, upper_bound, color=color, alpha=0.2)

ax2.set_ylim(-160, -80)
ax2.set_yticks(np.arange(-160, -80 + 1, 20))  # chia mỗi 20 đơn vị
ax2.set_xlabel("Training Episodes", fontsize=26)
ax2.legend(fontsize=26)
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=20)

# Thêm nhãn trục y chung căn giữa
fig.text(0.04, 0.5, 'Reward for An Episode', va='center', rotation='vertical', fontsize=26)

# Căn chỉnh layout để không che nhãn trục y
plt.tight_layout(rect=[0.06, 0, 1, 1])
plt.show()
