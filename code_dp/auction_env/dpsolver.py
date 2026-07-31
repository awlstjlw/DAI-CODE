import numpy as np
from scipy.stats import binom
from typing import List, Dict, Optional
from data_generator import Group
from fast_optimizer import solve_analytical_p0_vectorized 

class CompleteDPSolver:
    def __init__(self,
                 groups: List[Group],
                 calc_group_sizes: List[int],
                 T: int,
                 delta: float = 0.9,
                 epsilon: float = 0.1):
        self.groups = groups
        self.calc_group_sizes = calc_group_sizes
        self.T = T
        self.delta = delta
        self.epsilon = epsilon
        
        self.tmp_epsilon = np.full(T, epsilon, dtype=np.float64)
        
        self.g0_size, self.g1_size = calc_group_sizes
        self.SW = np.zeros((T + 1, self.g0_size + 1, self.g1_size + 1, 2), dtype=np.float64)
        
        self.optimal_p0 = np.zeros((T, self.g0_size + 1, self.g1_size + 1), dtype=np.float64)

        self.trans_mats = {
            0: self._precompute_group_transitions(0, self.g0_size),
            1: self._precompute_group_transitions(1, self.g1_size)
        }

    def _build_binom_matrix(self, n: int, p01: float, p11: float) -> np.ndarray:
        mat = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            probs_s1_s1 = binom.pmf(np.arange(i + 1), i, p11)
            probs_s0_s1 = binom.pmf(np.arange(n - i + 1), n - i, p01)
            probs_total = np.convolve(probs_s1_s1, probs_s0_s1)
            mat[i, :len(probs_total)] = probs_total
        return mat

    def _precompute_group_transitions(self, group_idx: int, size: int):
        group = self.groups[group_idx]
        mat0 = group.matrix0
        mat1 = group.matrix1
        
        T_idle = self._build_binom_matrix(size, mat0[0, 1], mat0[1, 1])
        
        if size == 0:
            return {'idle': T_idle, 'win_s0': None, 'win_s1': None}

        T_rest = self._build_binom_matrix(size - 1, mat0[0, 1], mat0[1, 1])
        
        T_win_s0 = np.zeros((size + 1, size + 1))
        
        p_w_00, p_w_01 = mat1[0, 0], mat1[0, 1]
        for i in range(size):
            T_win_s0[i, :] += p_w_00 * np.pad(T_rest[i, :], (0, 1))
            T_win_s0[i, :] += p_w_01 * np.pad(T_rest[i, :], (1, 0))

        T_win_s1 = np.zeros((size + 1, size + 1))
        
        p_w_10, p_w_11 = mat1[1, 0], mat1[1, 1]
        for i in range(1, size + 1):
            row_rest = T_rest[i - 1, :]
            T_win_s1[i, :] += p_w_10 * np.pad(row_rest, (0, 1))
            T_win_s1[i, :] += p_w_11 * np.pad(row_rest, (1, 0))
            
        return {'idle': T_idle, 'win_s0': T_win_s0, 'win_s1': T_win_s1}

    def solve(self) -> Dict[tuple, float]:
        v0_s0, v0_s1 = self.groups[0].valuations[0], self.groups[0].valuations[1]
        v1_s0, v1_s1 = self.groups[1].valuations[0], self.groups[1].valuations[1]

        for t in range(self.T - 1, -1, -1):
            next_welfare = self.SW[t + 1]
            nw_0 = next_welfare[..., 0]
            nw_1 = next_welfare[..., 1]
            
            def compute_exp_future(T0, T1):
                exp_0 = T0 @ nw_0 @ T1.T
                exp_1 = T0 @ nw_1 @ T1.T
                return exp_0 * self.delta, exp_1 * self.delta

            T0, T1 = self.trans_mats[0], self.trans_mats[1]
            
            fv_idle_0, fv_idle_1 = compute_exp_future(T0['idle'], T1['idle'])
            
            welfare_if_g0_wins_g0 = fv_idle_0.copy()
            welfare_if_g0_wins_g1 = fv_idle_1.copy()
            
            if self.g0_size > 0:
                fv_0, fv_1 = compute_exp_future(T0['win_s0'], T1['idle'])
                curr_0 = fv_0 + v0_s0
                mask = np.arange(self.g0_size + 1) < self.g0_size
                
                update = (curr_0 > welfare_if_g0_wins_g0) & mask[:, None]
                welfare_if_g0_wins_g0[update] = curr_0[update]
                welfare_if_g0_wins_g1[update] = fv_1[update]
                
                fv_0, fv_1 = compute_exp_future(T0['win_s1'], T1['idle'])
                curr_0 = fv_0 + v0_s1
                mask = np.arange(self.g0_size + 1) > 0
                
                update = (curr_0 > welfare_if_g0_wins_g0) & mask[:, None]
                welfare_if_g0_wins_g0[update] = curr_0[update]
                welfare_if_g0_wins_g1[update] = fv_1[update]

            welfare_if_g1_wins_g0 = fv_idle_0.copy()
            welfare_if_g1_wins_g1 = fv_idle_1.copy()
            
            if self.g1_size > 0:
                fv_0, fv_1 = compute_exp_future(T0['idle'], T1['win_s0'])
                curr_1 = fv_1 + v1_s0
                mask = np.arange(self.g1_size + 1) < self.g1_size
                
                update = (curr_1 > welfare_if_g1_wins_g1) & mask[None, :]
                welfare_if_g1_wins_g1[update] = curr_1[update]
                welfare_if_g1_wins_g0[update] = fv_0[update]
                
                fv_0, fv_1 = compute_exp_future(T0['idle'], T1['win_s1'])
                curr_1 = fv_1 + v1_s1
                mask = np.arange(self.g1_size + 1) > 0
                
                update = (curr_1 > welfare_if_g1_wins_g1) & mask[None, :]
                welfare_if_g1_wins_g1[update] = curr_1[update]
                welfare_if_g1_wins_g0[update] = fv_0[update]

            p0_grid = np.zeros_like(welfare_if_g0_wins_g0) + 0.5
            valid_mask = np.ones_like(p0_grid, dtype=bool)
            
            if self.g0_size == 0:
                p0_grid[:] = 0.0 
                valid_mask[:] = False
            elif self.g1_size == 0:
                p0_grid[:] = 1.0 
                valid_mask[:] = False
            
            if np.any(valid_mask):
                optimized_p0 = solve_analytical_p0_vectorized(
                    welfare_if_g0_wins_g0[valid_mask],
                    welfare_if_g1_wins_g0[valid_mask],
                    welfare_if_g0_wins_g1[valid_mask],
                    welfare_if_g1_wins_g1[valid_mask],
                    (self.g0_size, self.g1_size),
                    self.tmp_epsilon[t]
                )
                p0_grid[valid_mask] = optimized_p0
            
            p1_grid = 1.0 - p0_grid
            self.SW[t, ..., 0] = p0_grid * welfare_if_g0_wins_g0 + p1_grid * welfare_if_g1_wins_g0
            self.SW[t, ..., 1] = p0_grid * welfare_if_g0_wins_g1 + p1_grid * welfare_if_g1_wins_g1
            self.optimal_p0[t] = p0_grid

        result_dict = {}
        g0, g1 = self.calc_group_sizes
        for i in range(g0 + 1):
            for j in range(g1 + 1):
                state = (g0 - i, i, g1 - j, j)
                for t in range(self.T):
                    result_dict[(t, *state)] = float(self.optimal_p0[t, i, j])
                    
        return result_dict