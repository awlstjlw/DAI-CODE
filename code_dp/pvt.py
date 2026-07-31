import numpy as np
from scipy.stats import binom
from typing import List, Dict, Optional
from auction_env.data_generator import Group, Agent, OtherData
from auction_env.config import AuctionConfig
import time
import copy

class DynamicPivotAllocationManualCache:
    def __init__(self, groups: List[Group], agents: List[Agent], other_data: OtherData, config: AuctionConfig, rng: np.random.Generator):
        self.groups = groups
        self.agents = copy.deepcopy(agents)
        self.rng = rng
        
        self.T = config.T
        self.delta = config.DELTA
        self.group_sizes = other_data.group_sizes
        self.N0, self.N1 = self.group_sizes
        
        # DP tables for welfare calculation
        self.W = np.zeros((self.T + 1, self.N0 + 1, self.N1 + 1))
        self.W_group0 = np.zeros((self.T + 1, self.N0 + 1, self.N1 + 1))
        self.W_group1 = np.zeros((self.T + 1, self.N0 + 1, self.N1 + 1))
        
        # Policy table: 0:(G0,S0),1:(G0,S1),2:(G1,S0),3:(G1,S1),-1:no allocation
        self.policy = np.full((self.T, self.N0 + 1, self.N1 + 1), -1, dtype=int)

        # Welfare tables for no-i economy (VCG payment calculation)
        self.W_no_i = [
            np.zeros((self.T + 1, self.N0, self.N1 + 1)),
            np.zeros((self.T + 1, self.N0 + 1, self.N1))
        ]
        self.policy_no_i = [
            np.full((self.T, self.N0, self.N1 + 1), -1, dtype=int),
            np.full((self.T, self.N0 + 1, self.N1), -1, dtype=int)
        ]

        self.trans_mats = {
            0: self._precompute_group_transitions(0, self.N0),
            1: self._precompute_group_transitions(1, self.N1)
        }
        self.trans_mats_no_i = {
            0: self._precompute_group_transitions(0, self.N0 - 1),
            1: self._precompute_group_transitions(1, self.N1 - 1)
        }

        self.t_payment = [0.0] * self.T
        self.t_discounted_value = [0.0] * self.T
        self.t_discounted_value_group = [[0.0 for _ in range(self.T)] for _ in range(2)]
        self.discounted_value_group = [[0.0 for _ in range(self.T)] for _ in range(2)]
        self.t_discounted_expected_value_group = [[0.0 for _ in range(self.T)] for _ in range(2)]

    def _build_binom_matrix(self, n: int, p01: float, p11: float) -> np.ndarray:
        mat = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            probs_s1 = binom.pmf(np.arange(i + 1), i, p11)
            probs_s0 = binom.pmf(np.arange(n - i + 1), n - i, p01)
            probs_total = np.convolve(probs_s1, probs_s0)
            mat[i, :len(probs_total)] = probs_total
        return mat

    def _precompute_group_transitions(self, group_idx: int, size: int):
        if size < 0: return None
        group = self.groups[group_idx]
        
        T_idle = self._build_binom_matrix(size, group.matrix0[0, 1], group.matrix0[1, 1])
        
        if size == 0:
            return {'idle': T_idle}

        T_rest = self._build_binom_matrix(size - 1, group.matrix0[0, 1], group.matrix0[1, 1])
        
        T_win_s0 = np.zeros((size + 1, size + 1))
        T_win_s1 = np.zeros((size + 1, size + 1))
        
        # Win from S0
        p_w0, p_w1 = group.matrix1[0, 0], group.matrix1[0, 1]
        for i in range(size): 
            T_win_s0[i, :] += p_w0 * np.pad(T_rest[i, :], (0, 1))
            T_win_s0[i, :] += p_w1 * np.pad(T_rest[i, :], (1, 0))

        # Win from S1
        p_w0, p_w1 = group.matrix1[1, 0], group.matrix1[1, 1]
        for i in range(1, size + 1):
            row = T_rest[i - 1, :]
            T_win_s1[i, :] += p_w0 * np.pad(row, (0, 1))
            T_win_s1[i, :] += p_w1 * np.pad(row, (1, 0))
            
        return {'idle': T_idle, 'win_s0': T_win_s0, 'win_s1': T_win_s1}

    def backward_induction(self):
        start = time.time()
        
        v0_s0, v0_s1 = self.groups[0].valuations
        v1_s0, v1_s1 = self.groups[1].valuations
        
        for t in range(self.T - 1, -1, -1):
            V_next = self.W[t + 1]
            V_next_g0 = self.W_group0[t + 1]
            V_next_g1 = self.W_group1[t + 1]
            
            def calc_exp(T0, T1, V):
                return self.delta * (T0 @ V @ T1.T)

            # Idle transition
            E_idle = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['idle'], V_next)
            E_idle_g0 = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['idle'], V_next_g0)
            E_idle_g1 = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['idle'], V_next_g1)

            W_curr = E_idle.copy()
            W_g0_curr = E_idle_g0.copy()
            W_g1_curr = E_idle_g1.copy()
            policy_curr = np.full_like(self.policy[t], -1) 

            if self.N0 > 0:
                # G0-S0 Win
                E_win = calc_exp(self.trans_mats[0]['win_s0'], self.trans_mats[1]['idle'], V_next) + v0_s0
                E_win_g0 = calc_exp(self.trans_mats[0]['win_s0'], self.trans_mats[1]['idle'], V_next_g0) + v0_s0 
                E_win_g1 = calc_exp(self.trans_mats[0]['win_s0'], self.trans_mats[1]['idle'], V_next_g1)
                
                mask = (np.arange(self.N0 + 1) < self.N0)[:, None] & (E_win > W_curr)
                W_curr[mask], W_g0_curr[mask], W_g1_curr[mask] = E_win[mask], E_win_g0[mask], E_win_g1[mask]
                policy_curr[mask] = 0

                # G0-S1 Win
                E_win = calc_exp(self.trans_mats[0]['win_s1'], self.trans_mats[1]['idle'], V_next) + v0_s1
                E_win_g0 = calc_exp(self.trans_mats[0]['win_s1'], self.trans_mats[1]['idle'], V_next_g0) + v0_s1
                E_win_g1 = calc_exp(self.trans_mats[0]['win_s1'], self.trans_mats[1]['idle'], V_next_g1)

                mask = (np.arange(self.N0 + 1) > 0)[:, None] & (E_win > W_curr)
                W_curr[mask], W_g0_curr[mask], W_g1_curr[mask] = E_win[mask], E_win_g0[mask], E_win_g1[mask]
                policy_curr[mask] = 1

            if self.N1 > 0:
                # G1-S0 Win
                E_win = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['win_s0'], V_next) + v1_s0
                E_win_g0 = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['win_s0'], V_next_g0)
                E_win_g1 = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['win_s0'], V_next_g1) + v1_s0

                mask = (np.arange(self.N1 + 1) < self.N1)[None, :] & (E_win > W_curr)
                W_curr[mask], W_g0_curr[mask], W_g1_curr[mask] = E_win[mask], E_win_g0[mask], E_win_g1[mask]
                policy_curr[mask] = 2

                # G1-S1 Win
                E_win = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['win_s1'], V_next) + v1_s1
                E_win_g0 = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['win_s1'], V_next_g0)
                E_win_g1 = calc_exp(self.trans_mats[0]['idle'], self.trans_mats[1]['win_s1'], V_next_g1) + v1_s1

                mask = (np.arange(self.N1 + 1) > 0)[None, :] & (E_win > W_curr)
                W_curr[mask], W_g0_curr[mask], W_g1_curr[mask] = E_win[mask], E_win_g0[mask], E_win_g1[mask]
                policy_curr[mask] = 3

            self.W[t] = W_curr
            self.W_group0[t] = W_g0_curr
            self.W_group1[t] = W_g1_curr
            self.policy[t] = policy_curr

    def backward_induction_no_i(self):
        start = time.time()
        
        for g_miss in [0, 1]:
            mats0 = self.trans_mats_no_i[0] if g_miss == 0 else self.trans_mats[0]
            mats1 = self.trans_mats[1] if g_miss == 0 else self.trans_mats_no_i[1]
            W_tab = self.W_no_i[g_miss]
            P_tab = self.policy_no_i[g_miss]
            
            curr_N0 = self.N0 - 1 if g_miss == 0 else self.N0
            curr_N1 = self.N1 if g_miss == 0 else self.N1 - 1
            
            v0_s0, v0_s1 = self.groups[0].valuations
            v1_s0, v1_s1 = self.groups[1].valuations

            for t in range(self.T - 1, -1, -1):
                V_next = W_tab[t + 1]
                
                def calc_exp(T0, T1):
                    return self.delta * (T0 @ V_next @ T1.T)

                # Idle transition
                W_curr = calc_exp(mats0['idle'], mats1['idle'])
                policy_curr = np.full((curr_N0 + 1, curr_N1 + 1), -1, dtype=int)

                if curr_N0 > 0:
                    E_win = calc_exp(mats0['win_s0'], mats1['idle']) + v0_s0
                    mask = (np.arange(curr_N0+1) < curr_N0)[:, None] & (E_win > W_curr)
                    W_curr[mask] = E_win[mask]
                    policy_curr[mask] = 0

                    E_win = calc_exp(mats0['win_s1'], mats1['idle']) + v0_s1
                    mask = (np.arange(curr_N0+1) > 0)[:, None] & (E_win > W_curr)
                    W_curr[mask] = E_win[mask]
                    policy_curr[mask] = 1

                if curr_N1 > 0:
                    E_win = calc_exp(mats0['idle'], mats1['win_s0']) + v1_s0
                    mask = (np.arange(curr_N1+1) < curr_N1)[None, :] & (E_win > W_curr)
                    W_curr[mask] = E_win[mask]
                    policy_curr[mask] = 2
                    
                    E_win = calc_exp(mats0['idle'], mats1['win_s1']) + v1_s1
                    mask = (np.arange(curr_N1+1) > 0)[None, :] & (E_win > W_curr)
                    W_curr[mask] = E_win[mask]
                    policy_curr[mask] = 3

                W_tab[t] = W_curr
                P_tab[t] = policy_curr

    def compute_payment_optimized(self, t, g_id, s_id, current_s1_tuple):
        i, j = current_s1_tuple
        
        # Calculate no-i economy state
        i_no = i - 1 if (g_id == 0 and s_id == 1) else i
        j_no = j - 1 if (g_id == 1 and s_id == 1) else j
        
        act_code = self.policy_no_i[g_id][t, i_no, j_no]
        
        # Calculate term1
        term1 = 0.0
        if act_code != -1:
            act_g = 0 if act_code < 2 else 1
            act_s = act_code % 2
            term1 = self.groups[act_g].valuations[act_s]
        
        # Calculate term2
        mats0 = self.trans_mats_no_i[0] if g_id == 0 else self.trans_mats[0]
        mats1 = self.trans_mats[1] if g_id == 0 else self.trans_mats_no_i[1]
        V_next = self.W_no_i[g_id][t+1]
        
        def get_exp_val(ac, ii, jj):
            if ac == -1:
                vec0 = mats0['idle'][ii, :]
                vec1 = mats1['idle'][jj, :]
            elif ac == 0: vec0 = mats0['win_s0'][ii, :]; vec1 = mats1['idle'][jj, :]
            elif ac == 1: vec0 = mats0['win_s1'][ii, :]; vec1 = mats1['idle'][jj, :]
            elif ac == 2: vec0 = mats0['idle'][ii, :]; vec1 = mats1['win_s0'][jj, :]
            elif ac == 3: vec0 = mats0['idle'][ii, :]; vec1 = mats1['win_s1'][jj, :]
            
            return vec0 @ V_next @ vec1.T

        w_optimal = get_exp_val(act_code, i_no, j_no)
        w_actual = get_exp_val(-1, i_no, j_no)
        
        term2 = self.delta * (w_optimal - w_actual)
        
        return term1 + term2

    def real_auction(self):
        self.t_discounted_value_group = [[0.0 for _ in range(self.T)] for _ in range(2)]
        self.t_discounted_expected_value_group = [[0.0 for _ in range(self.T)] for _ in range(2)]

        for agent in self.agents:
            agent.state_probs = copy.deepcopy(self.groups[agent.label].init_prob)
            agent.cur_state = int(self.rng.choice(len(agent.state_probs), p=agent.state_probs))

        for t in range(self.T):
            counts = [0, 0, 0, 0]
            for a in self.agents:
                idx = 0 if a.label == 0 else 2
                counts[idx + a.cur_state] += 1
            
            s_idx = (counts[1], counts[3])
            act_code = self.policy[t][s_idx]
            
            action = None
            payment = 0.0
            winner_id = None
            
            self.t_discounted_value[t] = self.W[t][s_idx]
            self.discounted_value_group[0][t] = self.W_group0[t][s_idx]
            self.discounted_value_group[1][t] = self.W_group1[t][s_idx]
            self.t_discounted_expected_value_group[0][t] = self.W_group0[t][s_idx]
            self.t_discounted_expected_value_group[1][t] = self.W_group1[t][s_idx]

            if act_code != -1:
                g_id = 0 if act_code < 2 else 1
                s_id = act_code % 2
                action = (g_id, s_id)
                
                self.t_discounted_value_group[g_id][t] = self.groups[g_id].valuations[s_id]

                candidates = [a for a in self.agents if a.label == g_id and a.cur_state == s_id]
                if candidates:
                    winner_id = self.rng.choice([a.id for a in candidates])
                
                payment = self.compute_payment_optimized(t, g_id, s_id, s_idx)
            
            self.t_payment[t] = payment

            for agent in self.agents:
                won = (agent.id == winner_id)
                group = self.groups[agent.label]
                probs = group.matrix1[agent.cur_state] if won else group.matrix0[agent.cur_state]
                agent.state_probs = probs
                agent.cur_state = int(self.rng.choice(2, p=probs))

        for t in range(self.T-2, -1, -1):
            self.t_payment[t] += (self.delta * self.t_payment[t+1])
            for g_id in range(2):
                self.t_discounted_value_group[g_id][t] += (self.delta * self.t_discounted_value_group[g_id][t+1])
                

    def solve(self):
        self.backward_induction()
        self.backward_induction_no_i()
        self.real_auction()
