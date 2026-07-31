import numpy as np
from scipy.stats import binom
from typing import List, Dict, Optional
from auction_env.data_generator import Group, Agent, OtherData
from auction_env.config import AuctionConfig
import time
import copy

class DynamicPivotAllocationManualCache:
    """Dynamic pivot allocation solver for three groups."""
    def __init__(self, groups: List[Group], agents: List[Agent], other_data: OtherData, config: AuctionConfig, rng: np.random.Generator):
        self.groups = groups
        self.agents = copy.deepcopy(agents)
        self.rng = rng
        self.T = config.T
        self.delta = config.DELTA
        self.group_sizes = other_data.group_sizes
        self.N0, self.N1, self.N2 = self.group_sizes

        # Welfare tensor: [T+1, N0+1, N1+1, N2+1]
        self.W = np.zeros((self.T + 1, self.N0 + 1, self.N1 + 1, self.N2 + 1))

        # Expected welfare per group (used for VCG auxiliary calculation)
        self.W_groups = [np.zeros_like(self.W) for _ in range(3)]

        # Policy table: action codes 0-5 allocate (group, state), -1 is idle
        self.policy = np.full((self.T, self.N0 + 1, self.N1 + 1, self.N2 + 1), -1, dtype=int)

        self.W_no_i = []
        self.policy_no_i = []

        dims = [
            (self.N0 - 1, self.N1, self.N2),
            (self.N0, self.N1 - 1, self.N2),
            (self.N0, self.N1, self.N2 - 1)
        ]

        for d in dims:
            self.W_no_i.append(np.zeros((self.T + 1, d[0] + 1, d[1] + 1, d[2] + 1)))
            self.policy_no_i.append(np.full((self.T, d[0] + 1, d[1] + 1, d[2] + 1), -1, dtype=int))

        self.trans_mats = {
            i: self._precompute_group_transitions(i, self.group_sizes[i])
            for i in range(3)
        }
        self.trans_mats_no_i = {
            i: self._precompute_group_transitions(i, self.group_sizes[i] - 1)
            for i in range(3)
        }

        self.t_payment = [0.0] * self.T
        self.t_discounted_value = [0.0] * self.T
        self.t_discounted_value_group = [[0.0] * self.T for _ in range(3)]
        self.discounted_value_group = [[0.0] * self.T for _ in range(3)]
        self.t_discounted_expected_value_group = [[0.0] * self.T for _ in range(3)]

    def _build_binom_matrix(self, n, p01, p11):
        mat = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            probs_s1 = binom.pmf(np.arange(i + 1), i, p11)
            probs_s0 = binom.pmf(np.arange(n - i + 1), n - i, p01)
            probs_total = np.convolve(probs_s1, probs_s0)
            mat[i, :len(probs_total)] = probs_total
        return mat

    def _precompute_group_transitions(self, group_idx, size):
        if size < 0: return None
        group = self.groups[group_idx]
        T_idle = self._build_binom_matrix(size, group.matrix0[0, 1], group.matrix0[1, 1])
        if size == 0: return {'idle': T_idle}

        T_rest = self._build_binom_matrix(size - 1, group.matrix0[0, 1], group.matrix0[1, 1])
        T_win_s0 = np.zeros((size + 1, size + 1))
        T_win_s1 = np.zeros((size + 1, size + 1))

        p_w0, p_w1 = group.matrix1[0, 0], group.matrix1[0, 1]
        for i in range(size):
            T_win_s0[i, :] += p_w0 * np.pad(T_rest[i, :], (0, 1))
            T_win_s0[i, :] += p_w1 * np.pad(T_rest[i, :], (1, 0))

        p_w0, p_w1 = group.matrix1[1, 0], group.matrix1[1, 1]
        for i in range(1, size + 1):
            row = T_rest[i - 1, :]
            T_win_s1[i, :] += p_w0 * np.pad(row, (0, 1))
            T_win_s1[i, :] += p_w1 * np.pad(row, (1, 0))

        return {'idle': T_idle, 'win_s0': T_win_s0, 'win_s1': T_win_s1}

    def backward_induction(self):
        """Solve the full economy."""
        vals = [self.groups[i].valuations for i in range(3)]

        for t in range(self.T - 1, -1, -1):
            V_next = self.W[t + 1]
            V_next_groups = [self.W_groups[i][t + 1] for i in range(3)]

            # 3-group convolution: einsum('il, jm, kn, lmn -> ijk')
            def calc_exp(T0, T1, T2, Target_V):
                res = np.einsum('il, jm, kn, lmn -> ijk', T0, T1, T2, Target_V)
                return self.delta * res

            mats = [self.trans_mats[i]['idle'] for i in range(3)]
            E_idle = calc_exp(*mats, V_next)
            E_idle_groups = [calc_exp(*mats, V_next_groups[i]) for i in range(3)]

            W_curr = E_idle.copy()
            W_groups_curr = [g.copy() for g in E_idle_groups]
            policy_curr = np.full_like(self.policy[t], -1)

            for g_idx in range(3):
                g_size = self.group_sizes[g_idx]
                if g_size == 0: continue

                mats = [self.trans_mats[i]['idle'] for i in range(3)]
                mats[g_idx] = self.trans_mats[g_idx]['win_s0']

                E_win = calc_exp(*mats, V_next) + vals[g_idx][0]
                E_win_groups = [calc_exp(*mats, V_next_groups[i]) for i in range(3)]
                E_win_groups[g_idx] += vals[g_idx][0]

                dims = [self.N0+1, self.N1+1, self.N2+1]
                grids = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')

                # s0 count > 0 is equivalent to s1 count < group size
                mask_s0 = (grids[g_idx] < g_size) & (E_win > W_curr)

                if np.any(mask_s0):
                    W_curr[mask_s0] = E_win[mask_s0]
                    for i in range(3): W_groups_curr[i][mask_s0] = E_win_groups[i][mask_s0]
                    policy_curr[mask_s0] = g_idx * 2  # actions 0, 2, 4

                mats[g_idx] = self.trans_mats[g_idx]['win_s1']
                E_win = calc_exp(*mats, V_next) + vals[g_idx][1]
                E_win_groups = [calc_exp(*mats, V_next_groups[i]) for i in range(3)]
                E_win_groups[g_idx] += vals[g_idx][1]

                mask_s1 = (grids[g_idx] > 0) & (E_win > W_curr)
                if np.any(mask_s1):
                    W_curr[mask_s1] = E_win[mask_s1]
                    for i in range(3): W_groups_curr[i][mask_s1] = E_win_groups[i][mask_s1]
                    policy_curr[mask_s1] = g_idx * 2 + 1  # actions 1, 3, 5

            self.W[t] = W_curr
            for i in range(3): self.W_groups[i][t] = W_groups_curr[i]
            self.policy[t] = policy_curr

    def backward_induction_no_i(self):
        """Solve the three marginal economies."""
        vals = [self.groups[i].valuations for i in range(3)]

        for g_miss in range(3):
            W_tab = self.W_no_i[g_miss]
            P_tab = self.policy_no_i[g_miss]

            curr_sizes = list(self.group_sizes)
            curr_sizes[g_miss] -= 1

            for t in range(self.T - 1, -1, -1):
                V_next = W_tab[t + 1]

                def calc_exp(mats_list):
                    return self.delta * np.einsum('il, jm, kn, lmn -> ijk', *mats_list, V_next)

                # Mix full-economy and marginal-economy transition matrices
                base_mats = []
                for i in range(3):
                    if i == g_miss: base_mats.append(self.trans_mats_no_i[i]['idle'])
                    else: base_mats.append(self.trans_mats[i]['idle'])

                W_curr = calc_exp(base_mats)
                dims = [s + 1 for s in curr_sizes]
                policy_curr = np.full(dims, -1, dtype=int)
                grids = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij')

                for g_act in range(3):
                    if curr_sizes[g_act] == 0: continue

                    curr_mats = base_mats.copy()
                    tm = self.trans_mats_no_i if g_act == g_miss else self.trans_mats

                    curr_mats[g_act] = tm[g_act]['win_s0']
                    E_win = calc_exp(curr_mats) + vals[g_act][0]
                    mask = (grids[g_act] < curr_sizes[g_act]) & (E_win > W_curr)
                    W_curr[mask] = E_win[mask]
                    policy_curr[mask] = g_act * 2

                    curr_mats[g_act] = tm[g_act]['win_s1']
                    E_win = calc_exp(curr_mats) + vals[g_act][1]
                    mask = (grids[g_act] > 0) & (E_win > W_curr)
                    W_curr[mask] = E_win[mask]
                    policy_curr[mask] = g_act * 2 + 1

                W_tab[t] = W_curr
                P_tab[t] = policy_curr

    def compute_payment_optimized(self, t, g_id, s_id, s_tuple):
        """Calculate VCG payment for the winner."""
        # s_tuple: (s1_count_g0, s1_count_g1, s1_count_g2)
        s_counts = list(s_tuple)

        # If the winner came from S1, remove that S1 agent from the no-i economy
        if s_id == 1:
            s_counts[g_id] -= 1

        act_code = self.policy_no_i[g_id][t][tuple(s_counts)]

        # Value the marginal economy would assign to the next-best winner
        term1 = 0.0
        if act_code != -1:
            act_g = act_code // 2
            act_s = act_code % 2
            term1 = self.groups[act_g].valuations[act_s]

        V_next = self.W_no_i[g_id][t+1]

        def get_exp_val(code):
            mats = []
            for i in range(3):
                tm = self.trans_mats_no_i if i == g_id else self.trans_mats
                key = 'idle'
                if code != -1:
                    ag, as_ = code // 2, code % 2
                    if i == ag: key = 'win_s0' if as_ == 0 else 'win_s1'

                vec = tm[i][key][s_counts[i], :]
                mats.append(vec)

            return np.einsum('i, j, k, ijk -> ', mats[0], mats[1], mats[2], V_next)

        w_optimal = get_exp_val(act_code)
        w_actual = get_exp_val(-1)

        term2 = self.delta * (w_optimal - w_actual)
        return term1 + term2

    def real_auction(self, output_file="real_auction_optimized.txt"):
        with open(output_file, "w", encoding="utf-8") as f:
            self.t_discounted_value_group = [[0.0] * self.T for _ in range(3)]
            self.t_discounted_expected_value_group = [[0.0] * self.T for _ in range(3)]

            for agent in self.agents:
                agent.state_probs = copy.deepcopy(self.groups[agent.label].init_prob)
                agent.cur_state = int(self.rng.choice(len(agent.state_probs), p=agent.state_probs))

            for t in range(self.T):
                s1_counts = [0, 0, 0]
                for a in self.agents:
                    if a.cur_state == 1: s1_counts[a.label] += 1

                s_idx = tuple(s1_counts)
                act_code = self.policy[t][s_idx]

                self.t_discounted_value[t] = self.W[t][s_idx]
                for i in range(3):
                    self.discounted_value_group[i][t] = self.W_groups[i][t][s_idx]
                    self.t_discounted_expected_value_group[i][t] = self.W_groups[i][t][s_idx]

                action = None
                payment = 0.0
                winner_id = None

                if act_code != -1:
                    g_id = act_code // 2
                    s_id = act_code % 2
                    action = (g_id, s_id)
                    self.t_discounted_value_group[g_id][t] = self.groups[g_id].valuations[s_id]

                    candidates = [a for a in self.agents if a.label == g_id and a.cur_state == s_id]
                    if candidates: winner_id = self.rng.choice([a.id for a in candidates])

                    payment = self.compute_payment_optimized(t, g_id, s_id, s_idx)

                self.t_payment[t] = payment

                for agent in self.agents:
                    won = (agent.id == winner_id)
                    group = self.groups[agent.label]
                    probs = group.matrix1[agent.cur_state] if won else group.matrix0[agent.cur_state]
                    agent.cur_state = int(self.rng.choice(2, p=probs))

            for t in range(self.T-2, -1, -1):
                self.t_payment[t] += (self.delta * self.t_payment[t+1])
                for g_id in range(3):
                    self.t_discounted_value_group[g_id][t] += (self.delta * self.t_discounted_value_group[g_id][t+1])

    def solve(self):
        self.backward_induction()
        self.backward_induction_no_i()
        self.real_auction()
