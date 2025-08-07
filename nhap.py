import gymnasium as gym
import numpy as np
from collections import deque
from scipy.special import erf

class ENV_paper(gym.Env):
    def __init__(self, lambda_rate, D_max, xi, max_power, snr_feedback, harq_type):
        # Parameters
        self.lambda_rate = lambda_rate
        self.D_max = D_max
        self.xi = xi
        self.max_power = max_power
        self.snr_feedback = snr_feedback
        self.harq_type = harq_type
        self.n = 200  # Channel uses per slot
        self.T = int(10 / xi)  # Slots per episode
        self.Delta = 20 * max_power  # Penalty coefficient
        self.beta = 16  # Penalty exponent

        # State space
        state_dim = 4 if not snr_feedback else 5
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Action space
        self.action_space = gym.spaces.Box(
            low=np.array([0.1, 0.001]), 
            high=np.array([10.0, max_power]), 
            dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # State variables
        self.q_t = 0  # Queue length
        self.A_t = np.random.poisson(self.lambda_rate)  # Arrivals
        self.d_t = 0  # Delay violations
        self.k = 0  # Transmission count (0=new)
        self.t = 0  # Timestep
        self.gamma_k = 0.0  # Residual SNR
        
        # HARQ states
        self.initial_R = None
        self.initial_power = None
        self.previous_snrs = []
        self.arrival_history = deque(maxlen=self.D_max)  # For PODD
        self.arrival_history.append(self.A_t)
        
        return self._get_state(), {}

    def step(self, action):
        R_t, p_t = action
        R_t = np.clip(R_t, 0.1, 10.0)
        p_t = np.clip(p_t, 0.001, self.max_power)
        self.t += 1

        # KÊNH TRUYỀN
        """
        Tổn hao truyền dẫn: h_l
        Trải rộng chùm tia: h_bl
        Nhiễu loạn: h_t
        => Hệ số kênh h = h_l * h_bl * h_t
        """
        # Tham số
        distance = 30 # khoảng cách từ máy phát tới máy thu (m)
        extinction_coe = {'pure sea': 0.05, 'clear ocean': 0.15, 'coastal water': 0.31, 'habor': 2.17} # (m^-1)
        a  = 0.1 # bán kính máy thu (m)
        W_L = 4 # beam width (m)
        r = 0 # khoảng cách tù tâm chùm beam tới thấu kính thu (m)
        sigma_T2 = (10**-3)**2 # Nhiễu loạn mạnh
        mu_T = sigma_T2 / (-2)
        # Tổn hao truyền dẫn
        h_l = np.exp(-distance * extinction_coe['pure sea'])
        # Trải rộng chùm tia
        v = np.sqrt(np.pi) * a / (np.sqrt(2) * W_L)
        A_0 = erf(v)**2
        W_Leq = W_L**2 * (np.sqrt(np.pi) * erf(v)) / (2 * v * np.exp(-v**2))
        h_bl = A_0 * np.exp(-2 * r**2 / W_Leq)
        # Nhiễu loạn
        X = np.random.lognormal(mean=mu_T, sigma=np.sqrt(sigma_T2))
        h_t = np.exp(X)
        # Hệ số kênh
        h = h_l * h_t * h_bl
        snr = p_t * (np.abs(h)**2)

        # New transmission
        if self.k == 0:
            self.initial_R = R_t
            self.initial_power = p_t
            target_R = R_t
            success = np.log2(1 + snr) >= target_R
        # Retransmission
        else:
            target_R = self.initial_R  # Use initial rate
            
            # CC-HARQ without SNR feedback
            if self.harq_type == 'CC' and not self.snr_feedback:
                accumulated_snr = sum(self.previous_snrs) + snr
                success = np.log2(1 + accumulated_snr) >= target_R
            
            # CC-HARQ with SNR feedback
            elif self.harq_type == 'CC' and self.snr_feedback:
                accumulated_snr = sum(self.previous_snrs) + snr
                success = np.log2(1 + accumulated_snr) >= target_R
            
            # IR-HARQ (requires SNR feedback)
            elif self.harq_type == 'IR' and self.snr_feedback:
                accumulated_mutual_info = sum([np.log2(1+s) for s in self.previous_snrs]) + np.log2(1+snr)
                success = accumulated_mutual_info >= target_R

        # Service rate
        S_t = self.n * target_R if success else 0

        # Update queue
        q_tmp = max(self.q_t + self.A_t - S_t, 0)
        q_th = sum(self.arrival_history)  # PODD threshold
        
        # Check delay violation
        delay_violation = q_tmp > q_th
        if delay_violation:
            self.d_t += 1
            w_t = self._calculate_penalty()
            reward = -p_t - w_t
        else:
            reward = -p_t

        # Apply PODD
        self.q_t = min(q_tmp, q_th)
        
        # Generate next arrival
        self.A_t = np.random.poisson(self.lambda_rate)
        self.arrival_history.append(self.A_t)

        # Update HARQ state
        if success:
            self.k = 0
            self.gamma_k = 0.0
            self.previous_snrs = []
        else:
            self.k += 1
            self.previous_snrs.append(snr)
            
            # Update residual SNR for feedback
            if self.snr_feedback:
                if self.harq_type == 'CC':
                    prev_sum = sum(self.previous_snrs[:-1]) if self.k > 1 else 0
                    self.gamma_k = max(2**target_R - 1 - prev_sum, 0)
                elif self.harq_type == 'IR':
                    prev_snrs = self.previous_snrs[:-1] if self.k > 1 else []
                    product = np.prod([1 + s for s in prev_snrs]) if prev_snrs else 1.0
                    self.gamma_k = max(2**target_R / product - 1, 0)

        # Terminal check
        terminated = self.t >= self.T
        return self._get_state(), reward, terminated, False, {}

    def _get_state(self):
        state = [self.q_t, self.A_t, self.d_t, self.k]
        if self.snr_feedback:
            state.append(self.gamma_k)
        return np.array(state, dtype=np.float32)

    def _calculate_penalty(self):
        target_violations = self.T * self.xi
        if self.d_t <= target_violations:
            return self.Delta * (self.d_t / target_violations) ** self.beta
        return self.Delta
