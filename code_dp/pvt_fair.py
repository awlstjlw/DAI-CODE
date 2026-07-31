from math import gamma
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.special import comb
import random, time, copy, os
import torch 

from auction_env.data_generator import Group, Agent, OtherData, generate_data
from auction_env.config import AuctionConfig
from auction_env.dpsolver import CompleteDPSolver
from pvt import DynamicPivotAllocationManualCache


class FairnessRealAuction:
    def __init__(self, groups: List[Group], agents: List[Agent],
                 other_data: OtherData, config: AuctionConfig, rng: np.random.Generator,
                 precomputed_rnn_p0: Optional[Dict[tuple[int, int, int], float]] = None):
        
        self.groups, self.other_data, self.config = groups, other_data, config
        self.agents = copy.deepcopy(agents)

        self.T, self.delta, self.epsilon = config.T, config.DELTA, config.EPSILON
        self.rng = rng

        self.real_dp_p0 : Dict[tuple[int, int, int], float] = {}
        self.real_rnn_p0 : Dict[tuple[int, int, int], float] = precomputed_rnn_p0 if precomputed_rnn_p0 is not None else {}
        self.use_precomputed_rnn = (precomputed_rnn_p0 is not None)

        self.dp_calc_SW_table = None
        self.rnn_calc_SW_table = None

        self.statistic_dp_p0 = [0.0] * self.T
        self.statistic_rnn_p0 = [0.0] * self.T 

        self.dp_time_cost = 0.0
        self.rnn_time_cost = 0.0
        self.pvt_time_cost = 0.0

        self.agents_size = other_data.agents_size
        self.group_sizes = other_data.group_sizes
        self.calc_group_sizes = other_data.calc_group_sizes
        self.auc_group_sizes = other_data.auc_group_sizes
        
        self.agents_calc = copy.deepcopy(other_data.agents_calc)
        self.agents_auc = copy.deepcopy(other_data.agents_auc)

        self.auc_all_states_group = [self.auc_generate_states_group(i) for i in range(2)] #s0, s1
        self.auc_all_states_group_reduced = [self.auc_generate_states_group_reduced(i) for i in range(2)] #s1
        self.auc_all_states_group_no_i = [self.auc_generate_states_group_no_i(i) for i in range(2)] #s0, s1
        self.auc_all_states_group_no_i_reduced = [self.auc_generate_states_group_no_i_reduced(i) for i in range(2)] #s1
        self.calc_all_states = self.calc_generate_states() #calc group's g0s0,g0s1,g1s0,g1s1
        self.calc_all_states_reduced = self.calc_generate_states_reduced() #calc group's g0s1,g1s1

        self.SW_auc_dp_group:  Dict[tuple[int, int, int, int, int], float] = {} 
        self.policy_auc_dp_group: Dict[tuple[int, int, int, int, int], Optional[int]] = {} 
        self.SW_auc_dp_group_no_i: Dict[tuple[int, int, int, int, int], float] = {} 
        self.policy_auc_dp_group_no_i: Dict[tuple[int, int, int, int, int], Optional[int]] = {} 

        self.SW_auc_rnn_group:  Dict[tuple[int, int, int, int, int], float] = {} 
        self.policy_auc_rnn_group: Dict[tuple[int, int, int, int, int], Optional[int]] = {} 
        self.SW_auc_rnn_group_no_i: Dict[tuple[int, int, int, int, int], float] = {} 
        self.policy_auc_rnn_group_no_i: Dict[tuple[int, int, int, int, int], Optional[int]] = {} 

        self.pvt_real_social_welfare = [0.0] * self.T
        self.pvt_real_payment = [0.0] * self.T
        self.dp_real_social_welfare = [0.0] * self.T
        self.dp_real_payment = [0.0] * self.T
        self.rnn_real_social_welfare = [0.0] * self.T
        self.rnn_real_payment = [0.0] * self.T

        self.pvt_real_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        self.dp_real_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        self.rnn_real_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        
        self.pvt_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        self.dp_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        self.rnn_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        
        self.pvt_only_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        self.dp_only_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        self.rnn_only_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]

        self.dp_calc_only_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]
        self.rnn_calc_only_expected_social_welfare_group = [[0.0] * (self.T + 1) for _ in range(2)]

        self.transition_cache: Dict[tuple, Dict] = {}
        self.transition_cache_group: Dict[tuple[tuple[int, int, int], Optional[int]], Dict[tuple[int, int], float]] = {}

        # Precompute transition matrices
        self.calc_g0_trans_matrix = self._build_transition_matrix(self.groups[0].matrix0, self.calc_group_sizes[0])
        self.calc_g1_trans_matrix = self._build_transition_matrix(self.groups[1].matrix0, self.calc_group_sizes[1])
        
        self.calc_g0_win0_matrix = self._build_winner_matrix_calc(0, 0, self.calc_group_sizes[0]) # G0 wins S0
        self.calc_g0_win1_matrix = self._build_winner_matrix_calc(0, 1, self.calc_group_sizes[0]) # G0 wins S1
        self.calc_g1_win0_matrix = self._build_winner_matrix_calc(1, 0, self.calc_group_sizes[1]) # G1 wins S0
        self.calc_g1_win1_matrix = self._build_winner_matrix_calc(1, 1, self.calc_group_sizes[1]) # G1 wins S1

        self.auc_trans_matrices_no_action = [
            self._build_transition_matrix(self.groups[i].matrix0, self.auc_group_sizes[i])
            for i in range(2)
        ]
        self.auc_trans_matrices_no_action_no_i = [
            self._build_transition_matrix(self.groups[i].matrix0, self.auc_group_sizes[i] - 1)
            for i in range(2)
        ]

    def auc_generate_states_group(self, group_id: int):
        g = self.auc_group_sizes[group_id]
        return [(g - s1, s1) for s1 in range(g + 1)]
    
    def auc_generate_states_group_reduced(self, group_id: int):
        g = self.auc_group_sizes[group_id]
        return [s1 for s1 in range(g + 1)]

    def auc_generate_states_group_no_i(self, group_id: int):
        g = self.auc_group_sizes[group_id] - 1
        return [(g - s1, s1) for s1 in range(g + 1)]
    
    def auc_generate_states_group_no_i_reduced(self, group_id: int):
        g = self.auc_group_sizes[group_id] - 1
        return [s1 for s1 in range(g + 1)]

    def calc_generate_states(self) -> List[tuple[int, ...]]:
        g0, g1 = self.calc_group_sizes
        return [(g0 - g0s1, g0s1, g1 - g1s1, g1s1)
                for g0s1 in range(g0 + 1)
                for g1s1 in range(g1 + 1)]
    
    def calc_generate_states_reduced(self):
        g0, g1 = self.calc_group_sizes
        return [(g0s1, g1s1) 
                for g0s1 in range(g0 + 1)
                for g1s1 in range(g1 + 1)]

    def compute_single_group_transition(self, s0_count: int, s1_count: int,
                                        matrix: np.ndarray) -> Dict[tuple[int, int], float]:
        if s0_count + s1_count == 0:
            return {(0, 0): 1.0}
        distribution = {}
        p_00, p_01 = matrix[0, 0], matrix[0, 1]
        p_10, p_11 = matrix[1, 0], matrix[1, 1]
        for s0_to_0 in range(s0_count + 1):
            s0_to_1 = s0_count - s0_to_0
            prob_s0 = comb(s0_count, s0_to_0) * (p_00 ** s0_to_0) * (p_01 ** s0_to_1) if s0_count > 0 else 1.0
            for s1_to_0 in range(s1_count + 1):
                s1_to_1 = s1_count - s1_to_0
                prob_s1 = comb(s1_count, s1_to_0) * (p_10 ** s1_to_0) * (p_11 ** s1_to_1) if s1_count > 0 else 1.0
                new_s0, new_s1 = s0_to_0 + s1_to_0, s0_to_1 + s1_to_1
                distribution[(new_s0, new_s1)] = distribution.get((new_s0, new_s1), 0.0) + prob_s0 * prob_s1
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}
        return distribution

    def compute_winner_group_transition(self, s0_count: int, s1_count: int,
                                        winner_state: int, group: Group) -> Dict[tuple[int, int], float]:
        if (winner_state == 0 and s0_count == 0) or (winner_state == 1 and s1_count == 0):
            return {(s0_count, s1_count): 1.0}
        distribution = {}
        winner_probs = group.matrix1[winner_state, :]

        if winner_state == 0:
            rest_s0, rest_s1 = s0_count - 1, s1_count
        else:
            rest_s0, rest_s1 = s0_count, s1_count - 1
        rest_distribution = self.compute_single_group_transition(rest_s0, rest_s1, group.matrix0)
        for winner_next in [0, 1]:
            p_winner = winner_probs[winner_next]
            for (rest_new_s0, rest_new_s1), p_rest in rest_distribution.items():
                if winner_next == 0:
                    final_s0, final_s1 = rest_new_s0 + 1, rest_new_s1
                else:
                    final_s0, final_s1 = rest_new_s0, rest_new_s1 + 1
                prob = p_winner * p_rest
                distribution[(final_s0, final_s1)] = distribution.get((final_s0, final_s1), 0.0) + prob
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}
        return distribution

    def compute_exact_transition_probabilities_group(self, state_ingroup: tuple[int, int, int],
                                                     action_state_ingroup: Optional[int]) -> Dict[tuple[int, int], float]:
        cache_key = (state_ingroup, action_state_ingroup)
        if cache_key in self.transition_cache_group:
            return self.transition_cache_group[cache_key]
        group_id, s0, s1 = state_ingroup
        transition_probs = {}
        
        if action_state_ingroup is None:
            dist = self.compute_single_group_transition(s0, s1, self.groups[group_id].matrix0)
            for (new_s0, new_s1), p in dist.items():
                transition_probs[(new_s0, new_s1)] = transition_probs.get((new_s0, new_s1), 0.0) + p
        else:
            dist = self.compute_winner_group_transition(s0, s1, action_state_ingroup, self.groups[group_id])
            for (new_s0, new_s1),p in dist.items():
                transition_probs[(new_s0, new_s1)] = transition_probs.get((new_s0, new_s1), 0.0) + p

        total = sum(transition_probs.values())
        if abs(total - 1.0) > 1e-6:
            transition_probs = {s: p / total for s, p in transition_probs.items()}

        self.transition_cache_group[cache_key] = transition_probs
        return transition_probs

    def compute_exact_transition_probabilities(self, state: tuple[int, int, int, int], 
                                              action: Optional[tuple[int, int]]) -> Dict[tuple[int, int, int, int], float]:
        cache_key = (state, action)
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]
        
        g0s0, g0s1, g1s0, g1s1 = state
        transition_probs = {}
        
        if action is None:
            g0_dist = self.compute_single_group_transition(g0s0, g0s1, self.groups[0].matrix0)
            g1_dist = self.compute_single_group_transition(g1s0, g1s1, self.groups[1].matrix0)
            
            for (new_g0s0, new_g0s1), p0 in g0_dist.items():
                for (new_g1s0, new_g1s1), p1 in g1_dist.items():
                    next_state = (new_g0s0, new_g0s1, new_g1s0, new_g1s1)
                    if p0 * p1 > 1e-10:
                        transition_probs[next_state] = p0 * p1
        else:
            group_id, state_id = action
            if group_id == 0: 
                g0_dist = self.compute_winner_group_transition(g0s0, g0s1, state_id, self.groups[0])
                g1_dist = self.compute_single_group_transition(g1s0, g1s1, self.groups[1].matrix0)
            else:
                g0_dist = self.compute_single_group_transition(g0s0, g0s1, self.groups[0].matrix0)
                g1_dist = self.compute_winner_group_transition(g1s0, g1s1, state_id, self.groups[1])
            
            for (new_g0s0, new_g0s1), p0 in g0_dist.items():
                for (new_g1s0, new_g1s1), p1 in g1_dist.items():
                    next_state = (new_g0s0, new_g0s1, new_g1s0, new_g1s1)
                    if p0 * p1 > 1e-10:
                        transition_probs[next_state] = p0 * p1
        
        total = sum(transition_probs.values())
        if abs(total - 1.0) > 1e-6:
            transition_probs = {s: p/total for s, p in transition_probs.items()}
        
        self.transition_cache[cache_key] = transition_probs
        return transition_probs

    def _build_transition_matrix(self, base_matrix: np.ndarray, group_size: int) -> np.ndarray:
        dim = group_size + 1
        M = np.zeros((dim, dim))
        p_00, p_01 = base_matrix[0, 0], base_matrix[0, 1]
        p_10, p_11 = base_matrix[1, 0], base_matrix[1, 1]

        for s1_count in range(dim):
            s0_count = group_size - s1_count
            dist_s0 = np.array([comb(s0_count, k) * (p_01**k) * (p_00**(s0_count-k)) for k in range(s0_count + 1)])
            dist_s1 = np.array([comb(s1_count, k) * (p_11**k) * (p_10**(s1_count-k)) for k in range(s1_count + 1)])
            probs = np.convolve(dist_s0, dist_s1)
            total = np.sum(probs)
            if total > 0:
                probs = probs / total
            M[s1_count, :] = probs
        return M

    # [New] Build winner transition matrix specifically for Calc groups (where size doesn't reduce)
    def _build_winner_matrix_calc(self, group_id: int, winner_state: int, group_size: int) -> np.ndarray:
        dim = group_size + 1
        # Rest of the group evolves as normal (size - 1)
        if group_size - 1 < 0: # Case size=0 (should not happen in typical logic but for safety)
             return np.eye(1)
             
        M_rest = self._build_transition_matrix(self.groups[group_id].matrix0, group_size - 1)
        M_win = np.zeros((dim, dim))
        
        # Winner evolves based on Matrix1
        w_probs = self.groups[group_id].matrix1[winner_state, :] 
        
        for s1 in range(dim):
            s0 = group_size - s1
            # Check if this winner_state is possible in this configuration
            # e.g. winner_state=0 requires at least one s0
            if (winner_state == 0 and s0 > 0) or (winner_state == 1 and s1 > 0):
                rest_idx_s1 = s1 if winner_state == 0 else s1 - 1
                dist_rest = M_rest[rest_idx_s1, :] 
                
                # Winner becomes s0
                M_win[s1, :len(dist_rest)] += dist_rest * w_probs[0]
                # Winner becomes s1
                M_win[s1, 1:len(dist_rest)+1] += dist_rest * w_probs[1]
        
        row_sums = M_win.sum(axis=1, keepdims=True)
        # Avoid division by zero for impossible states
        np.divide(M_win, row_sums, out=M_win, where=(row_sums > 0))
        return M_win

    def _build_winner_matrix(self, group_id: int, winner_state: int, is_no_i: bool) -> Tuple[np.ndarray, np.ndarray]:
        total_size = self.auc_group_sizes[group_id] - (1 if is_no_i else 0)
        dim = total_size + 1
        rest_size = total_size - 1
        
        if rest_size == self.auc_group_sizes[group_id] - 1:
             M_rest = self.auc_trans_matrices_no_action_no_i[group_id]
        else:
             M_rest = self._build_transition_matrix(self.groups[group_id].matrix0, rest_size)
        
        M_win = np.zeros((dim, dim))
        valid_mask = np.zeros(dim, dtype=bool)
        w_probs = self.groups[group_id].matrix1[winner_state, :] 
        
        for s1 in range(dim):
            s0 = total_size - s1
            if (winner_state == 0 and s0 > 0) or (winner_state == 1 and s1 > 0):
                valid_mask[s1] = True
                rest_idx_s1 = s1 if winner_state == 0 else s1 - 1
                dist_rest = M_rest[rest_idx_s1, :] 
                M_win[s1, :len(dist_rest)] += dist_rest * w_probs[0]
                M_win[s1, 1:len(dist_rest)+1] += dist_rest * w_probs[1]
        
        row_sums = M_win.sum(axis=1, keepdims=True)
        np.divide(M_win, row_sums, out=M_win, where=(row_sums > 0))
        return M_win, valid_mask

    def _vectorized_backward_step(self, SW_next_array: np.ndarray, p0_array_t: np.ndarray, 
                                  group_id: int, is_no_i: bool) -> Tuple[np.ndarray, np.ndarray]:
        E_future_calc = np.einsum('ij, kl, jlm -> ikm', 
                                  self.calc_g0_trans_matrix, 
                                  self.calc_g1_trans_matrix, 
                                  SW_next_array)
        
        M_auc_none = self.auc_trans_matrices_no_action_no_i[group_id] if is_no_i else self.auc_trans_matrices_no_action[group_id]
        E_future_no_alloc = np.einsum('xy, ij y -> ij x', M_auc_none, E_future_calc)
        val_no_alloc = self.delta * E_future_no_alloc

        best_values = val_no_alloc.copy()
        best_actions = np.full(best_values.shape, -1, dtype=int)
        
        if group_id == 0:
            prob_win_t = p0_array_t
        else:
            prob_win_t = 1.0 - p0_array_t
            
        prob_lose_t = 1.0 - prob_win_t

        for winner_state in [0, 1]:
            M_auc_win, valid_mask = self._build_winner_matrix(group_id, winner_state, is_no_i)
            E_future_alloc = np.einsum('xy, ij y -> ij x', M_auc_win, E_future_calc)
            
            current_val = self.groups[group_id].valuations[winner_state]
            term_alloc = current_val + self.delta * E_future_alloc
            
            total_val = prob_win_t * term_alloc + prob_lose_t * val_no_alloc
            
            mask = (total_val > best_values) & valid_mask.reshape(1, 1, -1)
            best_values = np.where(mask, total_val, best_values)
            best_actions = np.where(mask, winner_state, best_actions)

        return best_values, best_actions

    def _run_backward_optimized(self, prefix: str, is_no_i: bool):
        suffix = "_no_i" if is_no_i else ""
        SW_dict = getattr(self, f'SW_auc_{prefix}_group{suffix}')
        Policy_dict = getattr(self, f'policy_auc_{prefix}_group{suffix}')
        p0_source = self.real_dp_p0 if prefix == 'dp' else self.real_rnn_p0
        
        c0_dim = self.calc_group_sizes[0] + 1
        c1_dim = self.calc_group_sizes[1] + 1
        
        p0_full = np.zeros((self.T, c0_dim, c1_dim))
        for t in range(self.T):
            for g0s1 in range(c0_dim):
                for g1s1 in range(c1_dim):
                    p0_full[t, g0s1, g1s1] = p0_source.get((t, g0s1, g1s1), 0.5)

        for group_id in range(2):
            auc_size = self.auc_group_sizes[group_id] - (1 if is_no_i else 0)
            auc_dim = auc_size + 1
            
            V_next = np.zeros((c0_dim, c1_dim, auc_dim))
            for g0 in range(c0_dim):
                for g1 in range(c1_dim):
                    for a in range(auc_dim):
                         SW_dict[(g0, g1, group_id, self.T, a)] = 0.0
            
            for t in range(self.T - 1, -1, -1):
                p0_t = p0_full[t, :, :].reshape(c0_dim, c1_dim, 1)
                V_curr, Pol_curr = self._vectorized_backward_step(V_next, p0_t, group_id, is_no_i)
                
                it = np.nditer([V_curr, Pol_curr], flags=['multi_index'])
                for val, act in it:
                    idx = it.multi_index 
                    key = (idx[0], idx[1], group_id, t, idx[2])
                    SW_dict[key] = float(val)
                    Policy_dict[key] = int(act) if act != -1 else None
                V_next = V_curr

    # [New] Independent Policy Evaluation for "Calc-Only" Expected Welfare
    def _compute_calc_only_calc_SW_table(self, p0_source: Dict) -> np.ndarray:
        """Compute V(t, g0s1, g1s1, group_idx) for calc groups under a fixed policy."""
        c0_dim = self.calc_group_sizes[0] + 1
        c1_dim = self.calc_group_sizes[1] + 1
        
        p0_full = np.zeros((self.T, c0_dim, c1_dim))
        for t in range(self.T):
            for g0s1 in range(c0_dim):
                for g1s1 in range(c1_dim):
                    p0_full[t, g0s1, g1s1] = p0_source.get((t, g0s1, g1s1), 0.5)
        
        # V[t, g0, g1, group_idx]
        V = np.zeros((self.T + 1, c0_dim, c1_dim, 2))
        
        v0_s0, v0_s1 = self.groups[0].valuations[0], self.groups[0].valuations[1]
        v1_s0, v1_s1 = self.groups[1].valuations[0], self.groups[1].valuations[1]
        
        for t in range(self.T - 1, -1, -1):
            V_next = V[t + 1] # shape (c0, c1, 2)
            p0 = p0_full[t]   # shape (c0, c1)
            p1 = 1.0 - p0
            
            # --- If G0 Wins ---
            # 1. G0 wins S0
            E_next_g0_s0 = np.einsum('ij, kl, jlm -> ikm', 
                                     self.calc_g0_win0_matrix, 
                                     self.calc_g1_trans_matrix, 
                                     V_next)
            reward_g0_s0 = np.zeros((c0_dim, c1_dim, 2))
            mask_g0_s0 = np.arange(c0_dim) < (c0_dim - 1) # s0 count > 0
            reward_g0_s0[mask_g0_s0, :, 0] = v0_s0
            Val_g0_wins_s0 = self.delta * E_next_g0_s0 + reward_g0_s0

            # 2. G0 wins S1
            E_next_g0_s1 = np.einsum('ij, kl, jlm -> ikm', 
                                     self.calc_g0_win1_matrix, 
                                     self.calc_g1_trans_matrix, 
                                     V_next)
            reward_g0_s1 = np.zeros((c0_dim, c1_dim, 2))
            mask_g0_s1 = np.arange(c0_dim) > 0 # s1 count > 0
            reward_g0_s1[mask_g0_s1, :, 0] = v0_s1
            Val_g0_wins_s1 = self.delta * E_next_g0_s1 + reward_g0_s1
            
            # [FIXED G0 SELECTION using np.where]
            total_val_s0 = Val_g0_wins_s0[..., 0] + Val_g0_wins_s0[..., 1]
            total_val_s1 = Val_g0_wins_s1[..., 0] + Val_g0_wins_s1[..., 1]
            pick_s1_g0 = (total_val_s1 > total_val_s0) # (c0, c1)
            
            # Base choice
            Val_g0_wins = np.where(pick_s1_g0[..., None], Val_g0_wins_s1, Val_g0_wins_s0)
            
            # Enforce constraints (S0 must exist to pick S0, S1 must exist to pick S1)
            # mask_g0_s0: (c0,) -> broadcast to (c0, c1, 1)
            # Case: Only S0 valid (S1=0, S0>0) -> mask_g0_s0 & ~mask_g0_s1
            force_s0_mask_g0 = (mask_g0_s0 & ~mask_g0_s1)[:, None] # (c0, 1)
            Val_g0_wins = np.where(force_s0_mask_g0[..., None], Val_g0_wins_s0, Val_g0_wins)
            
            # Case: Only S1 valid (S0=0, S1>0)
            force_s1_mask_g0 = (mask_g0_s1 & ~mask_g0_s0)[:, None] # (c0, 1)
            Val_g0_wins = np.where(force_s1_mask_g0[..., None], Val_g0_wins_s1, Val_g0_wins)
            
            
            # --- If G1 Wins ---
            # 1. G1 wins S0
            E_next_g1_s0 = np.einsum('ij, kl, jlm -> ikm', 
                                     self.calc_g0_trans_matrix, 
                                     self.calc_g1_win0_matrix, 
                                     V_next)
            reward_g1_s0 = np.zeros((c0_dim, c1_dim, 2))
            mask_g1_s0 = np.arange(c1_dim) < (c1_dim - 1)
            reward_g1_s0[:, mask_g1_s0, 1] = v1_s0
            Val_g1_wins_s0 = self.delta * E_next_g1_s0 + reward_g1_s0
            
            # 2. G1 wins S1
            E_next_g1_s1 = np.einsum('ij, kl, jlm -> ikm', 
                                     self.calc_g0_trans_matrix, 
                                     self.calc_g1_win1_matrix, 
                                     V_next)
            reward_g1_s1 = np.zeros((c0_dim, c1_dim, 2))
            mask_g1_s1 = np.arange(c1_dim) > 0
            reward_g1_s1[:, mask_g1_s1, 1] = v1_s1
            Val_g1_wins_s1 = self.delta * E_next_g1_s1 + reward_g1_s1
            
            # [FIXED G1 SELECTIcON using np.where for proper broadcasting]
            total_val_s0_g1 = Val_g1_wins_s0[..., 0] + Val_g1_wins_s0[..., 1]
            total_val_s1_g1 = Val_g1_wins_s1[..., 0] + Val_g1_wins_s1[..., 1]
            pick_s1_g1 = (total_val_s1_g1 > total_val_s0_g1) # (c0, c1)
            
            # Base choice
            Val_g1_wins = np.where(pick_s1_g1[..., None], Val_g1_wins_s1, Val_g1_wins_s0)
            
            # Constraints for G1 (Masks are on axis 1)
            # mask_g1_s0: (c1,) -> broadcast to (c0, c1, 1)
            
            # Case: Only S0 valid (S1=0, S0>0) -> mask_g1_s0 & ~mask_g1_s1
            force_s0_mask_g1 = (mask_g1_s0 & ~mask_g1_s1)[None, :] # (1, c1)
            Val_g1_wins = np.where(force_s0_mask_g1[..., None], Val_g1_wins_s0, Val_g1_wins)
            
            # Case: Only S1 valid (S0=0, S1>0)
            force_s1_mask_g1 = (mask_g1_s1 & ~mask_g1_s0)[None, :] # (1, c1)
            Val_g1_wins = np.where(force_s1_mask_g1[..., None], Val_g1_wins_s1, Val_g1_wins)
            
            # --- Combine with Policy p0 ---
            V[t] = p0[..., None] * Val_g0_wins + p1[..., None] * Val_g1_wins
            
        return V

    def pvt(self):
        start_t = time.perf_counter()
        solver = DynamicPivotAllocationManualCache(self.groups, self.agents, self.other_data, self.config, self.rng)
        solver.solve()
        self.pvt_time_cost = time.perf_counter() - start_t
        for t in range(self.T):
            self.pvt_real_social_welfare[t] = solver.t_discounted_value[t]
            self.pvt_real_payment[t] = solver.t_payment[t]
            for g_id in range(2):
                self.pvt_real_social_welfare_group[g_id][t] = solver.t_discounted_value_group[g_id][t]
                self.pvt_expected_social_welfare_group[g_id][t] = solver.discounted_value_group[g_id][t]
                self.pvt_only_expected_social_welfare_group[g_id][t] = solver.discounted_value_group[g_id][t]

    def calc_dp(self):
        start_t = time.perf_counter() 
        
        solver = CompleteDPSolver(copy.deepcopy(self.groups), copy.deepcopy(self.calc_group_sizes),
                                  self.T, self.delta, self.epsilon)
        dp_p0 = solver.solve()
        
        for t in range(self.T):
            for state in self.calc_all_states:
                self.real_dp_p0[(t, state[1], state[3])] = dp_p0[(t, *state)]
        self.dp_time_cost = time.perf_counter() - start_t 
        self.dp_calc_SW_table = solver.SW 

    def calc_RNN(self):
        """Use the externally-provided precomputed_rnn_p0 table directly."""
        if not self.use_precomputed_rnn:
            raise RuntimeError(
                "calc_RNN() now requires `precomputed_rnn_p0` from the "
                "precomputed_rnn_p0 from the learned model forward pass; the TrueRNN fallback "
                "has been removed."
            )
        self.rnn_calc_SW_table = self._compute_calc_only_calc_SW_table(self.real_rnn_p0)

    def auc_backward(self, prefix: str):
        self._run_backward_optimized(prefix, is_no_i=False)

    def auc_backward_no_i(self, prefix: str):
        self._run_backward_optimized(prefix, is_no_i=True)

    def auc(self, prefix: str, num_simulations: int = 1):
        p0_arr = self.real_dp_p0 if prefix == 'dp' else self.real_rnn_p0
        p0_statistic_arr = self.statistic_dp_p0 if prefix == 'dp' else self.statistic_rnn_p0
        welfare_arr = getattr(self, f"{prefix}_real_social_welfare")
        payment_arr = getattr(self, f"{prefix}_real_payment")
        welfare_arr_group = getattr(self, f"{prefix}_real_social_welfare_group")
        expected_welfare_group = getattr(self, f"{prefix}_expected_social_welfare_group")
        only_expected_welfare_group = getattr(self, f"{prefix}_only_expected_social_welfare_group")
        
        calc_only_expected_arr = getattr(self, f"{prefix}_calc_only_expected_social_welfare_group")
        
        current_calc_SW_table = self.dp_calc_SW_table if prefix == 'dp' else self.rnn_calc_SW_table

        policy_group = getattr(self, f'policy_auc_{prefix}_group')
        SW_group = getattr(self, f'SW_auc_{prefix}_group')
        
        for sim in range(num_simulations):
            # Reset agent states for each simulation
            for agent in self.agents_auc:
                agent.state_probs = copy.deepcopy(self.groups[agent.label].init_prob) 
                agent.cur_state = int(self.rng.choice(len(agent.state_probs), p=agent.state_probs)) 
            for agent in self.agents_calc:
                agent.state_probs = copy.deepcopy(self.groups[agent.label].init_prob) 
                agent.cur_state = int(self.rng.choice(len(agent.state_probs), p=agent.state_probs)) 
            
            for t in range(self.T): 
                # 1. Count Calc group states
                calc_g0s1 = 0; calc_g1s1 = 0
                for agent in self.agents_calc:
                    if agent.label == 0 and agent.cur_state == 1:
                        calc_g0s1 += 1
                    elif agent.label == 1 and agent.cur_state == 1:
                        calc_g1s1 += 1
                
                if current_calc_SW_table is not None:
                    calc_only_expected_arr[0][t] = current_calc_SW_table[t, calc_g0s1, calc_g1s1, 0]
                    calc_only_expected_arr[1][t] = current_calc_SW_table[t, calc_g0s1, calc_g1s1, 1]

                # 2. Get policy p0
                p0 = p0_arr.get((t, calc_g0s1, calc_g1s1), 0.0)
                p0_statistic_arr[t] = p0
                
                # 3. Determine winning group
                winner_group_id = self.rng.choice([0, 1], p=[p0, 1.0 - p0])
                
                # 4. Count Auc group states
                auc_s0 = 0; auc_s1 = 0; auc_s0_no_win = 0; auc_s1_no_win = 0
                for agent in self.agents_auc:
                    if agent.label == winner_group_id:
                        if agent.cur_state == 0:
                            auc_s0 += 1
                        elif agent.cur_state == 1:
                            auc_s1 += 1
                    else:
                        if agent.cur_state == 0:
                            auc_s0_no_win += 1
                        elif agent.cur_state == 1:
                            auc_s1_no_win += 1
                
                # 5. Get allocation action
                action_state_id = policy_group.get((calc_g0s1, calc_g1s1, winner_group_id, t, auc_s1), None)
                winner_id_auc = None 
                if action_state_id is not None:
                    candidates = [agent.id for agent in self.agents_auc if agent.label == winner_group_id and agent.cur_state == action_state_id]
                    if candidates:
                        winner_id_auc = self.rng.choice(candidates)
                
                #Calc group must also select a winner and simulate allocation
                winner_id_calc = None
                if action_state_id is not None:
                    candidates_calc = [agent.id for agent in self.agents_calc if agent.label == winner_group_id and agent.cur_state == action_state_id]
                    if candidates_calc:
                        winner_id_calc = self.rng.choice(candidates_calc)
                
                # 6. Calculate results
                payment_arr[t] = self.compute_payment(winner_group_id, t, calc_g0s1, calc_g1s1, auc_s0, auc_s1, action_state_id, prefix)

                if action_state_id is not None:
                    welfare_arr_group[winner_group_id][t] = self.groups[winner_group_id].valuations[action_state_id]
                    welfare_arr[t] = self.groups[winner_group_id].valuations[action_state_id]
                
                # 7. Calculate expected welfare
                if winner_group_id == 0:
                    expected_welfare_group[0][t] = self.groups[0].valuations[action_state_id] if action_state_id is not None else 0.0
                    trans_calc = self.compute_exact_transition_probabilities((self.calc_group_sizes[0] - calc_g0s1, calc_g0s1, 
                                                                              self.calc_group_sizes[1] - calc_g1s1, calc_g1s1), None)
                    trans_auc = self.compute_exact_transition_probabilities_group((0, auc_s0, auc_s1), action_state_id)
                    term = 0
                    for next_state_calc, prob_calc in trans_calc.items():
                        for next_state_auc, prob_auc in trans_auc.items():
                            term += (prob_calc * prob_auc * SW_group.get((next_state_calc[1], next_state_calc[3], 0, t+1, next_state_auc[1]), 0.0))
                    expected_welfare_group[0][t] += (self.delta * term)

                    trans_auc_none = self.compute_exact_transition_probabilities_group((1, auc_s0_no_win, auc_s1_no_win), None)
                    term = 0
                    for next_state_calc, prob_calc in trans_calc.items():
                        for next_state_auc_none, prob_auc in trans_auc_none.items():
                            term += (prob_calc * prob_auc * SW_group.get((next_state_calc[1], next_state_calc[3], 1, t+1, next_state_auc_none[1]), 0.0))
                    expected_welfare_group[1][t] = self.delta * term

                    only_expected_welfare_group[0][t] = SW_group.get((calc_g0s1, calc_g1s1, 0, t, auc_s1), 0.0)
                    only_expected_welfare_group[1][t] = SW_group.get((calc_g0s1, calc_g1s1, 1, t, auc_s1_no_win), 0.0)

                else:
                    expected_welfare_group[1][t] = self.groups[1].valuations[action_state_id] if action_state_id is not None else 0.0
                    trans_calc = self.compute_exact_transition_probabilities((self.calc_group_sizes[0] - calc_g0s1, calc_g0s1, 
                                                                              self.calc_group_sizes[1] - calc_g1s1, calc_g1s1), None)
                    trans_auc = self.compute_exact_transition_probabilities_group((1, auc_s0, auc_s1), action_state_id)
                    term = 0
                    for next_state_calc, prob_calc in trans_calc.items():
                        for next_state_auc, prob_auc in trans_auc.items():
                            term += (prob_calc * prob_auc * SW_group.get((next_state_calc[1], next_state_calc[3], 1, t+1, next_state_auc[1]), 0.0))
                    expected_welfare_group[1][t] += (self.delta * term)

                    trans_auc_none = self.compute_exact_transition_probabilities_group((0, auc_s0_no_win, auc_s1_no_win), None)
                    term = 0
                    for next_state_calc, prob_calc in trans_calc.items():
                        for next_state_auc_none, prob_auc in trans_auc_none.items():
                            term += (prob_calc * prob_auc * SW_group.get((next_state_calc[1], next_state_calc[3], 0, t+1, next_state_auc_none[1]), 0.0))
                    expected_welfare_group[0][t] = self.delta * term

                    only_expected_welfare_group[0][t] = SW_group.get((calc_g0s1, calc_g1s1, 0, t, auc_s1_no_win), 0.0)
                    only_expected_welfare_group[1][t] = SW_group.get((calc_g0s1, calc_g1s1, 1, t, auc_s1), 0.0)

                # 8. State update
                # Update Calc group states
                for agent in self.agents_calc :
                    won = (agent.id == winner_id_calc)
                    P_row = (self.groups[agent.label].matrix1 if won else self.groups[agent.label].matrix0)[agent.cur_state]
                    agent.cur_state = int(self.rng.choice(len(P_row), p=P_row))
                    
                # Update Auc group states
                for agent in self.agents_auc: 
                    won = (agent.id == winner_id_auc)
                    P_row = (self.groups[agent.label].matrix1 if won else self.groups[agent.label].matrix0)[agent.cur_state]
                    agent.cur_state = int(self.rng.choice(len(P_row), p=P_row))

            # Calculate discounted welfare
            for t in range(self.T-2, -1, -1):
                welfare_arr[t] += (self.delta * welfare_arr[t+1])
                payment_arr[t] += (self.delta * payment_arr[t+1])
                for g_id in range(2):
                    welfare_arr_group[g_id][t] += (self.delta * welfare_arr_group[g_id][t+1])          

    def compute_payment(self, group_id: int, t: int, calc_g0s1: int, calc_g1s1: int, auc_s0: int, auc_s1: int,
                        action_state_id: Optional[int], prefix: str) -> float:
        if action_state_id is None:
            return 0.0
        auc_s0_no_i = auc_s0 - (1 if action_state_id == 0 else 0)
        auc_s1_no_i = auc_s1 - (1 if action_state_id == 1 else 0)
        policy_no_i = getattr(self, f'policy_auc_{prefix}_group_no_i')
        SW_no_i = getattr(self, f'SW_auc_{prefix}_group_no_i')
        action_state_id_no_i = policy_no_i.get((calc_g0s1, calc_g1s1, group_id, t, auc_s1_no_i), None)
        term1 = self.groups[group_id].valuations[action_state_id_no_i] if action_state_id_no_i is not None else 0.0 
        trans_calc = self.compute_exact_transition_probabilities((self.calc_group_sizes[0] - calc_g0s1, calc_g0s1, 
                                                                  self.calc_group_sizes[1] - calc_g1s1, calc_g1s1), None)
        trans_auc_with_i = self.compute_exact_transition_probabilities_group((group_id, auc_s0_no_i, auc_s1_no_i), None)
        W_group = 0.0
        for next_state_calc, prob_calc in trans_calc.items():
            for next_state_auc, prob_auc in trans_auc_with_i.items():
                W_group += (prob_calc * prob_auc * SW_no_i.get((next_state_calc[1], next_state_calc[3], group_id, t+1, next_state_auc[1]), 0.0))
        trans_auc_without_i = self.compute_exact_transition_probabilities_group((group_id, auc_s0_no_i, auc_s1_no_i), action_state_id_no_i)
        W_group_no_i = 0.0
        for next_state_calc, prob_calc in trans_calc.items():
            for next_state_auc, prob_auc in trans_auc_without_i.items():
                W_group_no_i += (prob_calc * prob_auc * SW_no_i.get((next_state_calc[1], next_state_calc[3], group_id, t+1, next_state_auc[1]), 0.0))
        term2 = self.delta * (W_group_no_i - W_group)
        return float(term1 + term2)

    def run(self):
        self.pvt() 
        self.calc_dp()
        self.calc_RNN()

        self.auc_backward('dp')
        self.auc_backward_no_i('dp')
        self.auc('dp')

        self.auc_backward('rnn')
        self.auc_backward_no_i('rnn')
        self.auc('rnn')

        return {
            'pvt_expected_social_welfare_group0': self.pvt_expected_social_welfare_group[0],
            'pvt_expected_social_welfare_group1': self.pvt_expected_social_welfare_group[1],
            'pvt_only_expected_social_welfare_group0': self.pvt_only_expected_social_welfare_group[0],
            'pvt_only_expected_social_welfare_group1': self.pvt_only_expected_social_welfare_group[1],
            'dp_expected_social_welfare_group0': self.dp_expected_social_welfare_group[0],
            'dp_expected_social_welfare_group1': self.dp_expected_social_welfare_group[1],
            'dp_only_expected_social_welfare_group0': self.dp_only_expected_social_welfare_group[0],
            'dp_only_expected_social_welfare_group1': self.dp_only_expected_social_welfare_group[1],
            'dp_calc_only_expected_social_welfare_group0': self.dp_calc_only_expected_social_welfare_group[0],
            'dp_calc_only_expected_social_welfare_group1': self.dp_calc_only_expected_social_welfare_group[1],
            'rnn_expected_social_welfare_group0': self.rnn_expected_social_welfare_group[0],
            'rnn_expected_social_welfare_group1': self.rnn_expected_social_welfare_group[1],
            'rnn_only_expected_social_welfare_group0': self.rnn_only_expected_social_welfare_group[0],
            'rnn_only_expected_social_welfare_group1': self.rnn_only_expected_social_welfare_group[1],
            'rnn_calc_only_expected_social_welfare_group0': self.rnn_calc_only_expected_social_welfare_group[0],
            'rnn_calc_only_expected_social_welfare_group1': self.rnn_calc_only_expected_social_welfare_group[1],

            'pvt_real_social_welfare_group0': self.pvt_real_social_welfare_group[0],
            'pvt_real_social_welfare_group1': self.pvt_real_social_welfare_group[1],
            'dp_real_social_welfare_group0': self.dp_real_social_welfare_group[0],
            'dp_real_social_welfare_group1': self.dp_real_social_welfare_group[1],
            'rnn_real_social_welfare_group0': self.rnn_real_social_welfare_group[0],
            'rnn_real_social_welfare_group1': self.rnn_real_social_welfare_group[1],

            'pvt_real_social_welfare': self.pvt_real_social_welfare,
            'pvt_real_payment': self.pvt_real_payment,

            'statistic_dp_p0': self.statistic_dp_p0,
            'dp_real_social_welfare': self.dp_real_social_welfare,
            'dp_real_payment': self.dp_real_payment,

            'statistic_rnn_p0': self.statistic_rnn_p0,
            'rnn_real_social_welfare': self.rnn_real_social_welfare,
            'rnn_real_payment': self.rnn_real_payment,
            
            # Return all computation time
            'dp_time_cost': self.dp_time_cost,
            'rnn_time_cost': self.rnn_time_cost,
            'pvt_time_cost': self.pvt_time_cost
        }

if __name__ == "__main__":
    config = AuctionConfig()
    global_rng = np.random.default_rng(config.SEED)
    groups, agents, other_data, config = generate_data()
    auction = FairnessRealAuction(groups, agents, other_data, config, global_rng)
    res = auction.run()