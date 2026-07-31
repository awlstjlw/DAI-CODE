from math import gamma, comb
import numpy as np
from typing import List, Dict, Optional, Tuple
import random, time, copy, os
import torch
from scipy.stats import binom

from auction_env.data_generator import Group, Agent, OtherData, generate_data
from auction_env.config import AuctionConfig
from auction_env.dpsolver_m3 import CompleteDPSolver
from pvt_m3 import DynamicPivotAllocationManualCache

class FairnessRealAuction:
    def __init__(self, groups: List[Group], agents: List[Agent],
                 other_data: OtherData, config: AuctionConfig, rng: np.random.Generator,
                 # [Params] Key: (t, g0s1, g1s1, g2s1) -> Value: (p0, p1, p2)
                 precomputed_rnn_p0: Optional[Dict[tuple[int, int, int, int], tuple[float, float, float]]] = None):
        
        self.groups, self.other_data, self.config = groups, other_data, config
        self.agents = copy.deepcopy(agents)
        self.T, self.delta, self.epsilon = config.T, config.DELTA, config.EPSILON
        self.rng = rng

        self.real_dp_p0 : Dict[tuple[int, int, int, int], tuple[float, float, float]] = {}
        self.real_rnn_p0 = precomputed_rnn_p0 if precomputed_rnn_p0 is not None else {}
        self.use_precomputed_rnn = (precomputed_rnn_p0 is not None)

        self.dp_calc_SW_table = None
        self.rnn_calc_SW_table = None

        self.statistic_dp_p0 = [(0.0, 0.0, 0.0)] * self.T
        self.statistic_rnn_p0 = [(0.0, 0.0, 0.0)] * self.T

        self.dp_time_cost = 0.0
        self.rnn_time_cost = 0.0

        self.group_sizes = other_data.group_sizes
        self.calc_group_sizes = other_data.calc_group_sizes # [N0, N1, N2]
        self.auc_group_sizes = other_data.auc_group_sizes
        
        self.agents_calc = copy.deepcopy(other_data.agents_calc)
        self.agents_auc = copy.deepcopy(other_data.agents_auc)

        self.calc_all_states = self.calc_generate_states()

        # Key: (g0s1, g1s1, g2s1, group_id, t, auc_s1)
        self.SW_auc_dp_group = {} 
        self.policy_auc_dp_group = {} 
        self.SW_auc_dp_group_no_i = {} 
        self.policy_auc_dp_group_no_i = {} 

        self.SW_auc_rnn_group = {} 
        self.policy_auc_rnn_group = {} 
        self.SW_auc_rnn_group_no_i = {} 
        self.policy_auc_rnn_group_no_i = {} 

        self.pvt_real_social_welfare = [0.0] * self.T
        self.pvt_real_payment = [0.0] * self.T
        self.dp_real_social_welfare = [0.0] * self.T
        self.dp_real_payment = [0.0] * self.T
        self.rnn_real_social_welfare = [0.0] * self.T
        self.rnn_real_payment = [0.0] * self.T

        def init_g_list(): return [[0.0] * (self.T + 1) for _ in range(3)]
        self.pvt_real_social_welfare_group = init_g_list()
        self.dp_real_social_welfare_group = init_g_list()
        self.rnn_real_social_welfare_group = init_g_list()
        
        self.pvt_expected_social_welfare_group = init_g_list()
        self.dp_expected_social_welfare_group = init_g_list()
        self.rnn_expected_social_welfare_group = init_g_list()
        
        self.pvt_only_expected_social_welfare_group = init_g_list()
        self.dp_only_expected_social_welfare_group = init_g_list()
        self.rnn_only_expected_social_welfare_group = init_g_list()

        self.dp_calc_only_expected_social_welfare_group = init_g_list()
        self.rnn_calc_only_expected_social_welfare_group = init_g_list()

        self.transition_cache_3g = {} 
        self.transition_cache_group = {}

        self.calc_trans_matrices = [
            self._build_transition_matrix(self.groups[i].matrix0, self.calc_group_sizes[i])
            for i in range(3)
        ]
        
        self.auc_trans_matrices_no_action = [
            self._build_transition_matrix(self.groups[i].matrix0, self.auc_group_sizes[i])
            for i in range(3)
        ]
        self.auc_trans_matrices_no_action_no_i = [
            self._build_transition_matrix(self.groups[i].matrix0, self.auc_group_sizes[i] - 1)
            for i in range(3)
        ]
        
        self.calc_win_matrices = {} # key: (group_id, win_state)
        for g_id in range(3):
            for s in range(2):
                self.calc_win_matrices[(g_id, s)] = self._build_winner_matrix_calc(g_id, s, self.calc_group_sizes[g_id])

    def calc_generate_states(self) -> List[tuple[int, ...]]:
        g0, g1, g2 = self.calc_group_sizes
        return [(g0 - i, i, g1 - j, j, g2 - k, k)
                for i in range(g0 + 1)
                for j in range(g1 + 1)
                for k in range(g2 + 1)]

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
            if total > 0: probs = probs / total
            M[s1_count, :] = probs
        return M

    def compute_single_group_transition(self, s0_count: int, s1_count: int, matrix: np.ndarray) -> Dict[tuple[int, int], float]:
        if s0_count + s1_count == 0: return {(0, 0): 1.0}
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
        if total > 0: distribution = {k: v / total for k, v in distribution.items()}
        return distribution

    def compute_winner_group_transition(self, s0_count: int, s1_count: int, winner_state: int, group: Group) -> Dict[tuple[int, int], float]:
        if (winner_state == 0 and s0_count == 0) or (winner_state == 1 and s1_count == 0):
            return {(s0_count, s1_count): 1.0}
        distribution = {}
        winner_probs = group.matrix1[winner_state, :]
        if winner_state == 0: rest_s0, rest_s1 = s0_count - 1, s1_count
        else: rest_s0, rest_s1 = s0_count, s1_count - 1
        
        rest_distribution = self.compute_single_group_transition(rest_s0, rest_s1, group.matrix0)
        for winner_next in [0, 1]:
            p_winner = winner_probs[winner_next]
            for (rs0, rs1), p_rest in rest_distribution.items():
                fs0, fs1 = (rs0 + 1, rs1) if winner_next == 0 else (rs0, rs1 + 1)
                distribution[(fs0, fs1)] = distribution.get((fs0, fs1), 0.0) + p_winner * p_rest
        
        total = sum(distribution.values())
        if total > 0: distribution = {k: v / total for k, v in distribution.items()}
        return distribution

    def compute_exact_transition_probabilities_3g(self, state: tuple, action: Optional[tuple]) -> Dict[tuple, float]:
        cache_key = (state, action)
        if cache_key in self.transition_cache_3g: return self.transition_cache_3g[cache_key]
        
        # state: (g0s0, g0s1, g1s0, g1s1, g2s0, g2s1)
        g_states = [(state[0], state[1]), (state[2], state[3]), (state[4], state[5])]
        dists = []
        for i in range(3):
            s0, s1 = g_states[i]
            if action is not None and action[0] == i:
                dists.append(self.compute_winner_group_transition(s0, s1, action[1], self.groups[i]))
            else:
                dists.append(self.compute_single_group_transition(s0, s1, self.groups[i].matrix0))
        
        transition_probs = {}
        for (n0s0, n0s1), p0 in dists[0].items():
            for (n1s0, n1s1), p1 in dists[1].items():
                for (n2s0, n2s1), p2 in dists[2].items():
                    joint_p = p0 * p1 * p2
                    if joint_p > 1e-10:
                        next_state = (n0s0, n0s1, n1s0, n1s1, n2s0, n2s1)
                        transition_probs[next_state] = joint_p
        
        total = sum(transition_probs.values())
        if abs(total - 1.0) > 1e-6:
            transition_probs = {s: p/total for s, p in transition_probs.items()}
            
        self.transition_cache_3g[cache_key] = transition_probs
        return transition_probs

    def compute_exact_transition_probabilities_group(self, state_ingroup: tuple, action_state_ingroup: Optional[int]) -> Dict[tuple, float]:
        cache_key = (state_ingroup, action_state_ingroup)
        if cache_key in self.transition_cache_group: return self.transition_cache_group[cache_key]
        group_id, s0, s1 = state_ingroup
        
        if action_state_ingroup is None:
            dist = self.compute_single_group_transition(s0, s1, self.groups[group_id].matrix0)
        else:
            dist = self.compute_winner_group_transition(s0, s1, action_state_ingroup, self.groups[group_id])
            
        transition_probs = {}
        for (ns0, ns1), p in dist.items():
            transition_probs[(ns0, ns1)] = transition_probs.get((ns0, ns1), 0.0) + p
        
        self.transition_cache_group[cache_key] = transition_probs
        return transition_probs
    
    def _build_winner_matrix_calc(self, group_id: int, winner_state: int, group_size: int) -> np.ndarray:
        """Build transition matrix for a Calc group when it wins."""
        dim = group_size + 1
        if group_size == 0: return np.eye(1)
        
        # Transition for the N-1 losers
        M_rest = self._build_transition_matrix(self.groups[group_id].matrix0, group_size - 1)
        M_win = np.zeros((dim, dim))
        
        p_win_to_1 = self.groups[group_id].matrix1[winner_state, 1]
        p_win_to_0 = self.groups[group_id].matrix1[winner_state, 0]
        
        for k in range(dim):
            n_s1 = k
            n_s0 = group_size - k
            # Validity check: if winner_state is 0, we need at least one 0. If 1, at least one 1.
            if (winner_state == 0 and n_s0 == 0) or (winner_state == 1 and n_s1 == 0):
                continue # Impossible state for this winner type
            
            rest_k = k if winner_state == 0 else k - 1
            
            dist_rest = M_rest[rest_k, :]
            
            M_win[k, :len(dist_rest)] += dist_rest * p_win_to_0
            M_win[k, 1:len(dist_rest)+1] += dist_rest * p_win_to_1
            
        # Normalize to handle numerical issues or impossible states (though logic should cover it)
        row_sums = M_win.sum(axis=1, keepdims=True)
        np.divide(M_win, row_sums, out=M_win, where=(row_sums > 0))
        return M_win

    def _build_winner_matrix(self, group_id: int, winner_state: int, is_no_i: bool) -> Tuple[np.ndarray, np.ndarray]:
        total_size = self.auc_group_sizes[group_id] - (1 if is_no_i else 0)
        dim = total_size + 1
        
        if total_size == self.auc_group_sizes[group_id]:
             M_rest = self.auc_trans_matrices_no_action_no_i[group_id]
        else:
             M_rest = self._build_transition_matrix(self.groups[group_id].matrix0, total_size - 1)
        
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

    def _vectorized_backward_step(self, SW_next_array: np.ndarray, p_probs_t: np.ndarray, 
                                  group_id: int, is_no_i: bool) -> Tuple[np.ndarray, np.ndarray]:
        E_future_calc = np.einsum('il, jm, kn, lmnx -> ijkx', 
                                  self.calc_trans_matrices[0], 
                                  self.calc_trans_matrices[1], 
                                  self.calc_trans_matrices[2],
                                  SW_next_array)
        
        M_auc_none = self.auc_trans_matrices_no_action_no_i[group_id] if is_no_i else self.auc_trans_matrices_no_action[group_id]
        E_future_no_alloc = np.einsum('xy, ijk y -> ijk x', M_auc_none, E_future_calc)
        val_no_alloc = self.delta * E_future_no_alloc

        total_exp_value = np.zeros_like(val_no_alloc)
        final_policy = np.full(val_no_alloc.shape, -1, dtype=int)

        for win_g in range(3):
            prob_g = p_probs_t[..., win_g].reshape(val_no_alloc.shape[:-1] + (1,))
            
            if win_g != group_id:
                total_exp_value += prob_g * val_no_alloc
            else:
                best_val_if_win = val_no_alloc.copy() 
                best_act_if_win = np.full(best_val_if_win.shape, -1, dtype=int)
                
                for winner_state in [0, 1]:
                    M_auc_win, valid_mask = self._build_winner_matrix(group_id, winner_state, is_no_i)
                    E_future_alloc = np.einsum('xy, ijk y -> ijk x', M_auc_win, E_future_calc)
                    term_alloc = self.groups[group_id].valuations[winner_state] + self.delta * E_future_alloc
                    
                    mask = (term_alloc > best_val_if_win) & valid_mask.reshape(1, 1, 1, -1)
                    best_val_if_win = np.where(mask, term_alloc, best_val_if_win)
                    best_act_if_win = np.where(mask, winner_state, best_act_if_win)
                
                total_exp_value += prob_g * best_val_if_win
                final_policy = best_act_if_win

        return total_exp_value, final_policy

    def _run_backward_optimized(self, prefix: str, is_no_i: bool):
        suffix = "_no_i" if is_no_i else ""
        SW_dict = getattr(self, f'SW_auc_{prefix}_group{suffix}')
        Policy_dict = getattr(self, f'policy_auc_{prefix}_group{suffix}')
        p0_source = self.real_dp_p0 if prefix == 'dp' else self.real_rnn_p0
        
        dims = [s+1 for s in self.calc_group_sizes]
        p_full = np.zeros((self.T, *dims, 3))
        for t in range(self.T):
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        p_full[t, i, j, k] = p0_source.get((t, i, j, k), (1/3, 1/3, 1/3))

        for group_id in range(3):
            auc_size = self.auc_group_sizes[group_id] - (1 if is_no_i else 0)
            auc_dim = auc_size + 1
            V_next = np.zeros((*dims, auc_dim))
            
            for t in range(self.T - 1, -1, -1):
                p_t = p_full[t]
                V_curr, Pol_curr = self._vectorized_backward_step(V_next, p_t, group_id, is_no_i)
                
                it = np.nditer(V_curr, flags=['multi_index'])
                for val in it:
                    idx = it.multi_index 
                    key = (idx[0], idx[1], idx[2], group_id, t, idx[3])
                    SW_dict[key] = float(val)
                    act = Pol_curr[idx]
                    Policy_dict[key] = int(act) if act != -1 else None
                V_next = V_curr
    
    def _compute_calc_only_calc_SW_table(self, p0_source: Dict) -> np.ndarray:
        """Compute V(t, g0s1, g1s1, g2s1, group_idx) for the Calc groups using p0_source."""
        dims = [s + 1 for s in self.calc_group_sizes] # [D0, D1, D2]
        
        # Prepare Policy Tensor: [T, D0, D1, D2, 3]
        p_full = np.zeros((self.T, *dims, 3))
        for t in range(self.T):
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        p_full[t, i, j, k] = p0_source.get((t, i, j, k), (1/3, 1/3, 1/3)) # default equal prob
        
        # V[t, g0, g1, g2, group_id] -> stores expected welfare for each group
        V = np.zeros((self.T + 1, *dims, 3))
        
        vals = [g.valuations for g in self.groups]

        for t in range(self.T - 1, -1, -1):
            V_next = V[t + 1]
            p_t = p_full[t]
            
            # p0_source gives only group win probabilities, so the winning group
            # chooses the state (0 or 1) that maximizes its own welfare.
            
            V_scenarios = np.zeros((*dims, 3, 3)) # (State..., Winner_ID, Benefit_Group_ID)
            
            for win_g in range(3):
                mats = [self.calc_trans_matrices[0], self.calc_trans_matrices[1], self.calc_trans_matrices[2]]
                best_outcome = np.full((*dims, 3), -np.inf)
                
                for s_act in [0, 1]:
                     mats[win_g] = self.calc_win_matrices[(win_g, s_act)]
                     
                     E_fut = np.einsum('il, jm, kn, lmnx -> ijkx', mats[0], mats[1], mats[2], V_next)
                     val_fut = self.delta * E_fut
                     
                     current_counts = np.arange(dims[win_g])
                     if s_act == 0:
                         valid_mask = (current_counts < self.calc_group_sizes[win_g])
                     else:
                         valid_mask = (current_counts > 0)
                     
                     slices = [slice(None)] * 3
                     slices[win_g] = valid_mask
                     full_mask = np.zeros(dims, dtype=bool)
                     full_mask[tuple(slices)] = True
                     
                     reward = np.zeros(3)
                     reward[win_g] = vals[win_g][s_act]
                     
                     total_val = val_fut + reward
                     
                     # "Best" maximizes the winner's own welfare.
                     better = (total_val[..., win_g] > best_outcome[..., win_g]) & full_mask
                     
                     best_outcome[better] = total_val[better]

                V_scenarios[..., win_g, :] = best_outcome

            # Average over winner: p_t shape (D0, D1, D2, 3), V_scenarios shape (D0, D1, D2, 3, 3)
            # Result V[t] shape (D0, D1, D2, 3)
            V[t] = np.einsum('ijkw, ijkwx -> ijkx', p_t, V_scenarios)

        return V

    def pvt(self):
        solver = DynamicPivotAllocationManualCache(self.groups, self.agents, self.other_data, self.config, self.rng)
        solver.solve()
        for t in range(self.T):
            self.pvt_real_social_welfare[t] = solver.t_discounted_value[t]
            self.pvt_real_payment[t] = solver.t_payment[t]
            for g_id in range(3):
                self.pvt_real_social_welfare_group[g_id][t] = solver.t_discounted_value_group[g_id][t]
                self.pvt_expected_social_welfare_group[g_id][t] = solver.discounted_value_group[g_id][t]
                self.pvt_only_expected_social_welfare_group[g_id][t] = solver.discounted_value_group[g_id][t]

    def calc_dp(self):
        start_t = time.perf_counter() 
        solver = CompleteDPSolver(copy.deepcopy(self.groups), copy.deepcopy(self.calc_group_sizes),
                                  self.T, self.delta, self.epsilon)
        dp_p0_dict = solver.solve()
        self.real_dp_p0 = dp_p0_dict
        self.dp_time_cost = time.perf_counter() - start_t 
        self.dp_calc_SW_table = solver.SW

    def calc_RNN(self):
        """Use the externally-provided precomputed_rnn_p0 table."""
        if not self.use_precomputed_rnn:
            raise RuntimeError(
                "calc_RNN() now requires `precomputed_rnn_p0` from the "
                "precomputed_rnn_p0 from the learned model forward pass; the TrueRNN fallback "
                "has been removed."
            )
        self.rnn_calc_SW_table = self._compute_calc_only_calc_SW_table(self.real_rnn_p0)


    def auc_backward(self, prefix: str): self._run_backward_optimized(prefix, False)
    def auc_backward_no_i(self, prefix: str): self._run_backward_optimized(prefix, True)

    def auc(self, prefix: str, num_simulations: int = 1, output_file: Optional[str] = None):
        if output_file is None: output_file = f"{prefix}_real_auction_fair_simulation.txt"
        
        p_src = self.real_dp_p0 if prefix == 'dp' else self.real_rnn_p0
        p_stat = self.statistic_dp_p0 if prefix == 'dp' else self.statistic_rnn_p0
        welfare_arr = getattr(self, f"{prefix}_real_social_welfare")
        payment_arr = getattr(self, f"{prefix}_real_payment")
        welfare_arr_group = getattr(self, f"{prefix}_real_social_welfare_group")
        expected_welfare_group = getattr(self, f"{prefix}_expected_social_welfare_group")
        only_expected_welfare_group = getattr(self, f"{prefix}_only_expected_social_welfare_group")

        calc_only_expected_arr = getattr(self, f"{prefix}_calc_only_expected_social_welfare_group")
        current_calc_SW_table = self.dp_calc_SW_table if prefix == 'dp' else self.rnn_calc_SW_table
        
        policy_group = getattr(self, f'policy_auc_{prefix}_group')
        SW_group = getattr(self, f'SW_auc_{prefix}_group')
        
        with open(output_file, "a", encoding="utf-8") as f:
            for sim in range(num_simulations):
                for a in self.agents_auc + self.agents_calc:
                    a.cur_state = int(self.rng.choice(len(self.groups[a.label].init_prob), p=self.groups[a.label].init_prob))
                
                for t in range(self.T):
                    c_s1 = [0, 0, 0]
                    for a in self.agents_calc: 
                        if a.cur_state == 1: c_s1[a.label] += 1
                    
                    if current_calc_SW_table is not None:
                        calc_only_expected_arr[0][t] = current_calc_SW_table[t, c_s1[0], c_s1[1], c_s1[2], 0]
                        calc_only_expected_arr[1][t] = current_calc_SW_table[t, c_s1[0], c_s1[1], c_s1[2], 1]
                        calc_only_expected_arr[2][t] = current_calc_SW_table[t, c_s1[0], c_s1[1], c_s1[2], 2]

                    probs = np.array(p_src.get((t, *c_s1), (1/3, 1/3, 1/3)), dtype=np.float64)
                    if probs.sum() > 0: probs /= probs.sum()
                    else: probs = np.array([1/3, 1/3, 1/3])
                    
                    p_stat[t] = tuple(probs)
                    win_g = self.rng.choice(3, p=probs)
                    auc_s1 = sum(1 for a in self.agents_auc if a.label == win_g and a.cur_state == 1)
                    
                    act_key = (*c_s1, win_g, t, auc_s1)
                    act_s = policy_group.get(act_key, None)
                    
                    win_id_auc = None; win_id_calc = None
                    if act_s is not None:
                        val = self.groups[win_g].valuations[act_s]
                        welfare_arr_group[win_g][t] = val
                        welfare_arr[t] = val
                        
                        cands_auc = [a.id for a in self.agents_auc if a.label == win_g and a.cur_state == act_s]
                        if cands_auc: win_id_auc = self.rng.choice(cands_auc)
                        cands_calc = [a.id for a in self.agents_calc if a.label == win_g and a.cur_state == act_s]
                        if cands_calc: win_id_calc = self.rng.choice(cands_calc)
                        
                        payment_arr[t] = self.compute_payment(win_g, t, c_s1, auc_s1, act_s, prefix)

                    trans_calc = self.compute_exact_transition_probabilities_3g(
                        (self.calc_group_sizes[0]-c_s1[0], c_s1[0], 
                         self.calc_group_sizes[1]-c_s1[1], c_s1[1],
                         self.calc_group_sizes[2]-c_s1[2], c_s1[2]), None
                    )
                    
                    for g_monitor in range(3):
                        only_expected_welfare_group[g_monitor][t] = SW_group.get((*c_s1, g_monitor, t, sum(1 for a in self.agents_auc if a.label==g_monitor and a.cur_state==1)), 0.0)
                        
                        is_winner = (g_monitor == win_g)
                        state_monitor = (g_monitor, sum(1 for a in self.agents_auc if a.label==g_monitor and a.cur_state==0), 
                                         sum(1 for a in self.agents_auc if a.label==g_monitor and a.cur_state==1))
                        
                        action_monitor = act_s if is_winner else None
                        trans_auc = self.compute_exact_transition_probabilities_group(state_monitor, action_monitor)
                        
                        term = 0.0
                        for next_c, p_c in trans_calc.items():
                             # next_c: (g0s0, g0s1, g1s0, g1s1, g2s0, g2s1)
                             # prefix: (g0s1, g1s1, g2s1, group_monitor, t+1)
                            next_prefix = (next_c[1], next_c[3], next_c[5], g_monitor, t+1)
                            for (ns0, ns1), p_a in trans_auc.items():
                                term += p_c * p_a * SW_group.get((*next_prefix, ns1), 0.0)
                        
                        immediate = self.groups[g_monitor].valuations[act_s] if (is_winner and act_s is not None) else 0.0
                        expected_welfare_group[g_monitor][t] = immediate + self.delta * term

                    for a in self.agents_auc:
                        mat = self.groups[a.label].matrix1 if (a.id == win_id_auc) else self.groups[a.label].matrix0
                        a.cur_state = int(self.rng.choice(2, p=mat[a.cur_state]))
                    for a in self.agents_calc:
                        mat = self.groups[a.label].matrix1 if (a.id == win_id_calc) else self.groups[a.label].matrix0
                        a.cur_state = int(self.rng.choice(2, p=mat[a.cur_state]))

                for t in range(self.T-2, -1, -1):
                    welfare_arr[t] += (self.delta * welfare_arr[t+1])
                    payment_arr[t] += (self.delta * payment_arr[t+1])
                    for g in range(3):
                         welfare_arr_group[g][t] += (self.delta * welfare_arr_group[g][t+1])

    def compute_payment(self, group_id: int, t: int, c_s1: List[int], auc_s1: int,
                        action: int, prefix: str) -> float:
        auc_s1_no_i = auc_s1 - (1 if action == 1 else 0)
        policy_no_i = getattr(self, f'policy_auc_{prefix}_group_no_i')
        SW_no_i = getattr(self, f'SW_auc_{prefix}_group_no_i')
        
        act_no_i = policy_no_i.get((*c_s1, group_id, t, auc_s1_no_i), None)
        term1 = self.groups[group_id].valuations[act_no_i] if act_no_i is not None else 0.0
        
        full_calc = (self.calc_group_sizes[0]-c_s1[0], c_s1[0], 
                     self.calc_group_sizes[1]-c_s1[1], c_s1[1],
                     self.calc_group_sizes[2]-c_s1[2], c_s1[2])
        trans_calc = self.compute_exact_transition_probabilities_3g(full_calc, None)
        
        auc_state_no_i = (group_id, (self.auc_group_sizes[group_id]-1)-auc_s1_no_i, auc_s1_no_i)
        
        trans_auc_actual = self.compute_exact_transition_probabilities_group(auc_state_no_i, None)
        trans_auc_optimal = self.compute_exact_transition_probabilities_group(auc_state_no_i, act_no_i)
        
        W_act = 0.0; W_opt = 0.0
        for next_c, p_c in trans_calc.items():
            prefix_next = (next_c[1], next_c[3], next_c[5], group_id, t+1)
            for (ns0, ns1), p_a in trans_auc_actual.items():
                W_act += p_c * p_a * SW_no_i.get((*prefix_next, ns1), 0.0)
            for (ns0, ns1), p_o in trans_auc_optimal.items():
                W_opt += p_c * p_o * SW_no_i.get((*prefix_next, ns1), 0.0)
                
        term2 = self.delta * (W_opt - W_act)
        return float(term1 + term2)

    def run(self):
        self.pvt(); self.calc_dp(); self.calc_RNN()
        self.auc_backward('dp'); self.auc_backward_no_i('dp'); self.auc('dp')
        self.auc_backward('rnn'); self.auc_backward_no_i('rnn'); self.auc('rnn')

        return {
            **{f'{m}_{k}_group{g}': getattr(self, f'{m}_{k}_group')[g]
            for m in ['pvt', 'dp', 'rnn']
            for k in ['expected_social_welfare', 'only_expected_social_welfare', 'real_social_welfare']
            for g in range(3)}, 

            **{f'{m}_calc_only_expected_social_welfare_group{g}': getattr(self, f'{m}_calc_only_expected_social_welfare_group')[g]
            for m in ['dp', 'rnn'] 
            for g in range(3)},

            **{f'{m}_{k}': getattr(self, f'{m}_{k}')
            for m in ['pvt', 'dp', 'rnn']
            for k in ['real_social_welfare', 'real_payment']},

            **{f'statistic_{m}_p0': getattr(self, f'statistic_{m}_p0') for m in ['dp', 'rnn']},
            **{f'{m}_time_cost': getattr(self, f'{m}_time_cost') for m in ['dp', 'rnn']}
        }
