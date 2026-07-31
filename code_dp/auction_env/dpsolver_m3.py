import numpy as np
from scipy.stats import binom
from typing import List, Dict
from .data_generator_m3 import Group
from .fast_optimizer_m3 import solve_optimization_3g_vectorized

class CompleteDPSolver:
    def __init__(self, groups, calc_group_sizes, T, delta, epsilon):
        self.groups = groups
        self.calc_group_sizes = calc_group_sizes
        self.T = T
        self.delta = delta
        self.epsilon = epsilon
        self.tmp_epsilon = np.full(T, epsilon, dtype=np.float64)
        
        self.g0, self.g1, self.g2 = calc_group_sizes
        
        # SW: [T+1, N0+1, N1+1, N2+1, 3] (last dim stores welfare for each group)
        self.SW = np.zeros((T + 1, self.g0 + 1, self.g1 + 1, self.g2 + 1, 3), dtype=np.float64)
        
        # Optimal P: [T, N0+1, N1+1, N2+1, 3] (last dim stores p0, p1, p2)
        self.optimal_p = np.zeros((T, self.g0 + 1, self.g1 + 1, self.g2 + 1, 3), dtype=np.float64)

        self.trans_mats = {
            0: self._precompute_group_transitions(0, self.g0),
            1: self._precompute_group_transitions(1, self.g1),
            2: self._precompute_group_transitions(2, self.g2)
        }

    def _build_binom_matrix(self, n, p01, p11):
        mat = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            probs_s1_s1 = binom.pmf(np.arange(i + 1), i, p11)
            probs_s0_s1 = binom.pmf(np.arange(n - i + 1), n - i, p01)
            probs_total = np.convolve(probs_s1_s1, probs_s0_s1)
            mat[i, :len(probs_total)] = probs_total
        return mat

    def _precompute_group_transitions(self, group_idx, size):
        group = self.groups[group_idx]
        mat0 = group.matrix0
        mat1 = group.matrix1
        T_idle = self._build_binom_matrix(size, mat0[0, 1], mat0[1, 1])
        if size == 0:
            return {'idle': T_idle, 'win_s0': None, 'win_s1': None}

        T_rest = self._build_binom_matrix(size - 1, mat0[0, 1], mat0[1, 1])
        T_win_s0 = np.zeros((size + 1, size + 1))
        p_w0, p_w1 = mat1[0, 0], mat1[0, 1]
        for i in range(size):
            T_win_s0[i, :] += p_w0 * np.pad(T_rest[i, :], (0, 1))
            T_win_s0[i, :] += p_w1 * np.pad(T_rest[i, :], (1, 0))

        T_win_s1 = np.zeros((size + 1, size + 1))
        p_w0, p_w1 = mat1[1, 0], mat1[1, 1]
        for i in range(1, size + 1):
            row_rest = T_rest[i - 1, :]
            T_win_s1[i, :] += p_w0 * np.pad(row_rest, (0, 1))
            T_win_s1[i, :] += p_w1 * np.pad(row_rest, (1, 0))
            
        return {'idle': T_idle, 'win_s0': T_win_s0, 'win_s1': T_win_s1}

    def solve(self):
        vals = [g.valuations for g in self.groups] 

        for t in range(self.T - 1, -1, -1):
            next_welfare = self.SW[t + 1] # (N0, N1, N2, 3)

            # T0(i,l), T1(j,m), T2(k,n), W(l,m,n,g) -> W(i,j,k,g)
            def compute_exp_future(T0, T1, T2):
                exp_val = np.einsum('il, jm, kn, lmng -> ijkg', T0, T1, T2, next_welfare)
                return exp_val * self.delta

            T0, T1, T2 = self.trans_mats[0], self.trans_mats[1], self.trans_mats[2]
            
            # Case 0: Idle
            fv_idle = compute_exp_future(T0['idle'], T1['idle'], T2['idle'])
            
            # Scenarios: [G0_Wins, G1_Wins, G2_Wins]
            welfare_scenarios = [fv_idle.copy() for _ in range(3)] 
            
            for win_g in range(3):
                g_size = self.calc_group_sizes[win_g]
                if g_size == 0: continue
                
                mats = [T0['idle'], T1['idle'], T2['idle']]
                
                # Try Win via S0
                mats[win_g] = self.trans_mats[win_g]['win_s0']
                fv_s0 = compute_exp_future(*mats)
                curr_s0 = fv_s0.copy()
                curr_s0[..., win_g] += vals[win_g][0]

                # Try Win via S1
                mats[win_g] = self.trans_mats[win_g]['win_s1']
                fv_s1 = compute_exp_future(*mats)
                curr_s1 = fv_s1.copy()
                curr_s1[..., win_g] += vals[win_g][1]

                # Maximize Total Welfare (Sum over last dimension)
                sum_s0 = np.sum(curr_s0, axis=-1)
                sum_s1 = np.sum(curr_s1, axis=-1)
                
                # Grid Masks
                shape_grid = [self.g0+1, self.g1+1, self.g2+1]
                grids = np.meshgrid(np.arange(shape_grid[0]), np.arange(shape_grid[1]), np.arange(shape_grid[2]), indexing='ij')
                s1_count = grids[win_g] # s1 count for the winning group
                
                mask_s0 = (s1_count < g_size)
                mask_s1 = (s1_count > 0)
                
                best_scenario = welfare_scenarios[win_g]
                best_sum = np.sum(best_scenario, axis=-1)

                improve_s0 = (sum_s0 > best_sum) & mask_s0
                best_scenario[improve_s0] = curr_s0[improve_s0]
                best_sum[improve_s0] = sum_s0[improve_s0]
                
                improve_s1 = (sum_s1 > best_sum) & mask_s1
                best_scenario[improve_s1] = curr_s1[improve_s1]
                best_sum[improve_s1] = sum_s1[improve_s1]
            
            # Optimization Step (Linear Program)
            w_g0_inputs = [welfare_scenarios[i][..., 0].flatten() for i in range(3)]
            w_g1_inputs = [welfare_scenarios[i][..., 1].flatten() for i in range(3)]
            w_g2_inputs = [welfare_scenarios[i][..., 2].flatten() for i in range(3)]
            
            res_p = solve_optimization_3g_vectorized(
                w_g0_inputs, w_g1_inputs, w_g2_inputs,
                (self.g0, self.g1, self.g2),
                self.tmp_epsilon[t]
            )
            
            grid_shape = (self.g0+1, self.g1+1, self.g2+1)
            p0_grid = res_p[:, 0].reshape(grid_shape)
            p1_grid = res_p[:, 1].reshape(grid_shape)
            p2_grid = res_p[:, 2].reshape(grid_shape)
            
            # SW = Expected Welfare based on probabilities
            self.SW[t] = (
                p0_grid[..., None] * welfare_scenarios[0] +
                p1_grid[..., None] * welfare_scenarios[1] +
                p2_grid[..., None] * welfare_scenarios[2]
            )
            
            self.optimal_p[t, ..., 0] = p0_grid
            self.optimal_p[t, ..., 1] = p1_grid
            self.optimal_p[t, ..., 2] = p2_grid

        # Package results
        result_dict = {}
        for t in range(self.T):
            for i in range(self.g0 + 1):
                for j in range(self.g1 + 1):
                    for k in range(self.g2 + 1):
                        probs = tuple(self.optimal_p[t, i, j, k])
                        # Key matches RNN input: (t, g0s1, g1s1, g2s1)
                        result_dict[(t, i, j, k)] = probs
        return result_dict