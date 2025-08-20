import gymnasium as gym
import numpy as np
from scipy.special import erf
from scipy.stats import f

class ENV_paper(gym.Env):
    def __init__(self, lambda_rate, D_max, xi, max_power, snr_feedback, harq_type):
        # System parameters
        self.lambda_rate = lambda_rate
        self.D_max = D_max
        self.xi = xi
        self.max_power = max_power
        self.snr_feedback = snr_feedback
        self.harq_type = harq_type  # 'CC', 'IR', or 'XP'
        self.n = 200
        self.T = int(10 / xi)
        self.Delta = 20 * max_power
        self.beta = 16
        self.R_bar = 10

        # State space: Use max dimension (8 if feedback, 7 if not)
        state_dim = 8 if snr_feedback else 7
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Action space: [coding_rate, transmit_power]
        self.action_space = gym.spaces.Box(
            low=np.array([0.1, 0.001]),
            high=np.array([self.R_bar, max_power]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.q_t = 0
        self.A_t = np.random.poisson(self.lambda_rate)
        self.d_t = 0
        self.k = 0
        self.t = 0
        self.gamma_k = 0.0
        self.R_sum = 0.0 if self.harq_type == 'XP' else 0.0  # Only for XP
        self.I_sum = 0.0 if self.harq_type == 'XP' else 0.0  # Only for XP
        self.h_prev = 1.0 if self.harq_type == 'XP' else 1.0  # Only for XP
        self.arrival_history = [self.A_t]
        self.previous_snrs = []
        self.previous_rates = []
        state = [self.q_t, self.A_t, self.d_t, self.k, self.gamma_k, self.R_sum, self.I_sum, self.h_prev]
        if not self.snr_feedback:
            state.pop(4)
        return np.array(state), {}

    def step(self, action):
        R_t, p_t = action
        self.t += 1
        R_t = np.clip(R_t, 0.1, self.R_bar)
        p_t = np.clip(p_t, 0.001, self.max_power)

        # Channel model (FSO)
        distance = 0.5
        coe = 1
        C2n = 1e-14
        fso_lambda = 1550e-9
        k = 2 * np.pi / fso_lambda
        r_a = 0.1
        w_z = 3.2
        FOV = 16.5e-3
        noise_var = 1e-9
        sigma_p2 = 0.2**2
        sigma_theta2 = (10**-3)**2
        h_l = np.exp(-distance * coe)
        Rytov_var = 1.23 * C2n * (k**(7/6)) * ((distance*1e3)**(11/6))
        sigma_lnS = (0.49 * Rytov_var**2) / (1 + 1.11 * Rytov_var**(6/5))**(7/6)
        sigma_lnL = (0.51 * Rytov_var**2) / (1 + 0.69 * Rytov_var**(6/5))**(5/6)
        a_f = 1 / (np.exp(sigma_lnS) - 1)
        b_f = 1 / (np.exp(sigma_lnL) - 1) + 2
        h_f = f.rvs(dfn=2*a_f, dfd=2*b_f)
        x_t = np.random.normal(0, sigma_p2)
        x_r = np.random.normal(0, sigma_p2)
        y_t = np.random.normal(0, sigma_p2)
        y_r = np.random.normal(0, sigma_p2)
        x_tr = x_t + x_r
        y_tr = y_t + y_r
        x_d = x_tr + distance * np.random.normal(0, sigma_theta2)
        y_d = y_tr + distance * np.random.normal(0, sigma_theta2)
        r_tr = np.sqrt(x_d**2 + y_d**2)
        v = np.sqrt(np.pi) * r_a / (np.sqrt(2) * w_z)
        A_0 = erf(v)**2
        w2_zeq = w_z**2 * (np.sqrt(np.pi) * erf(v)) / (2 * v * np.exp(-v**2))
        h_p = A_0 * np.exp(-2 * r_tr**2 / w2_zeq)
        theta_a = np.sqrt(2 * sigma_theta2)
        h_a = 1 if theta_a <= FOV else 0
        h = h_l * h_f * h_p * h_a  # FSO channel without Rayleigh fading
        self.h_prev = h if self.harq_type == 'XP' else 1.0  # Update only for XP
        snr = (10**(p_t/10)) * (np.abs(h)**2) / noise_var

        # HARQ Logic
        if self.k == 0:  # New transmission
            self.previous_rates = [R_t]
            self.R_sum = R_t if self.harq_type == 'XP' else R_t
            self.I_sum = np.log2(1 + snr) if self.harq_type == 'XP' else np.log2(1 + snr)
            self.previous_snrs = [snr]
        else:  # Retransmission
            if self.harq_type == 'XP':
                self.previous_rates.append(R_t)  # New rate for new packet
                self.R_sum += R_t
                self.I_sum += np.log2(1 + snr)
                self.previous_snrs.append(snr)
            elif self.harq_type == 'CC':
                self.previous_rates.append(self.previous_rates[0])
                self.I_sum = np.log2(1 + snr) 
                self.previous_snrs.append(snr)
            elif self.harq_type == 'IR':
                self.previous_rates.append(R_t)  # Redundancy rate
                self.I_sum += np.log2(1 + snr)
                self.previous_snrs.append(snr)

        # Set target rate
        if self.harq_type == 'CC' or self.harq_type == 'IR':
            target_R = self.previous_rates[0]  # Initial rate for CC/IR-HARQ
        else:  # XP-HARQ
            target_R = self.R_sum  # Accumulated rate for XP-HARQ

        # Check transmission success
        if self.harq_type == 'CC':
            success = np.log2(1 + sum(self.previous_snrs)) >= target_R
        else:  # IR or XP
            success = self.I_sum >= target_R

        # Calculate service rate
        S_t = self.n * target_R if success else 0

        # Update queue
        q_tmp = max(self.q_t + self.A_t - S_t, 0)
        q_th = sum(self.arrival_history[-self.D_max:]) if len(self.arrival_history) >= self.D_max else sum(self.arrival_history)
        delay_violation = q_tmp > q_th
        if delay_violation:
            self.d_t += 1
            w_t = self._calculate_penalty()
            reward = -p_t - w_t
        else:
            reward = -p_t
        self.q_t = min(q_tmp, q_th)

        # Update arrivals
        self.A_t = np.random.poisson(self.lambda_rate)
        self.arrival_history.append(self.A_t)
        if len(self.arrival_history) > self.D_max:
            self.arrival_history.pop(0)

        # Update transmission state
        if success or self.k >= self.D_max:
            self.k = 0
            self.gamma_k = 0.0
            self.R_sum = 0.0
            self.I_sum = 0.0
            self.previous_snrs = []
            self.previous_rates = []
            # if success:
            #     print(f"Cycle {self.t}: Success, decode m_1 to m_{self.k+1}")
            # else:
            #     print(f"Cycle {self.t}: Failed after k={self.D_max}, start new cycle with m_1")
        else:
            self.k += 1
            if self.snr_feedback:
                if self.harq_type == 'XP':
                    self.gamma_k = max(2**(self.R_sum - self.I_sum) - 1, 0)
                elif self.harq_type == 'CC':
                    accumulated_snr = sum(self.previous_snrs[:-1]) if self.k > 1 else 0
                    self.gamma_k = max(2**target_R - 1 - accumulated_snr, 0)
                elif self.harq_type == 'IR':
                    prev_snrs = self.previous_snrs[:-1] if self.k > 1 else []
                    product = np.prod([1 + s for s in prev_snrs]) if prev_snrs else 1
                    self.gamma_k = max(2**target_R / product - 1, 0)

        # Build next state
        state = [self.q_t, self.A_t, self.d_t, self.k]
        if self.snr_feedback:
            state.append(self.gamma_k)
        if self.harq_type == 'XP':
            state.extend([self.R_sum, self.I_sum, self.h_prev])
        else:
            # For IR/CC, pad with dummy values to match dimension
            state.extend([0.0, 0.0, 1.0])  # Dummy for R_sum, I_sum, h_prev
        state = np.array(state)

        # Episode termination
        terminated = self.t >= self.T
        truncated = False
        done = terminated or truncated

        # Info for debugging
        info = {
            'power': p_t,
            'rate': R_t,
            'success': success,
            'delay_violation': delay_violation,
            'total_delay_violations': self.d_t,
            'transmission_count': self.k,
            'R_sum': self.R_sum,
            'I_sum': self.I_sum
        }

        return state, reward, done, info

    def _calculate_penalty(self):
        target_violations = self.T * self.xi
        if self.d_t <= target_violations:
            return self.Delta * (self.d_t / target_violations) ** self.beta
        return self.Delta
