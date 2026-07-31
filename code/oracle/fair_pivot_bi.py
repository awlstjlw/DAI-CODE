"""Exact backward induction oracle for the equity-constrained dynamic pivot mechanism."""

import itertools
import faulthandler
import math
import numpy as np
import os
import warnings
from typing import List, Dict, Optional, Tuple, Union

faulthandler.enable()
from itertools import product

# scipy is optional for the linprog path (only needed for m=3,4 K=2);
# fall back gracefully if not present.
try:
    from scipy.optimize import linprog
    _HAS_LINPROG = True
except ImportError:
    _HAS_LINPROG = False


def enumerate_market_states(group_sizes: List[int], num_states: int) -> np.ndarray:
    """Enumerate all market states in flattened (S, m*K) format.

    Returns (S, m*K) int64 array.
    """
    if num_states < 2:
        raise ValueError(f"num_states must be >= 2, got {num_states}")

    def _gen(g_idx, current):
        if g_idx >= len(group_sizes):
            result.append(current.copy())
            return
        N = group_sizes[g_idx]
        _enumerate_one_group(N, num_states, 0, [N], current, g_idx, _gen, result)

    result = []
    _gen(0, [])
    return np.array(result, dtype=np.int64)


def _enumerate_one_group(N, K, state_idx, remaining_dist, current,
                          g_idx, _gen, result):
    """Recursively enumerate one group's state counts."""
    if state_idx == K - 1:
        current.extend(remaining_dist)
        _gen(g_idx + 1, current)
        current.pop()
        return
    remaining = remaining_dist[0]
    for c in range(remaining + 1):
        remaining_dist[0] = remaining - c
        current.append(c)
        _enumerate_one_group(N, K, state_idx + 1, remaining_dist,
                             current, g_idx, _gen, result)
        current.pop()
    remaining_dist[0] = remaining


def enumerate_market_states_K(group_sizes: List[int], num_states: int) -> np.ndarray:
    return enumerate_market_states(group_sizes, num_states)



def _compositions(n: int, k: int):
    """All k-tuples of non-negative integers summing to n."""
    if k == 0:
        if n == 0:
            yield ()
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def _binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))



def build_transition_matrix(matrix0: np.ndarray, group_size: int) -> np.ndarray:
    """
    K=2 idle transition matrix for one group (rows = current #S1, cols = next #S1).
    """
    assert matrix0.shape == (2, 2)
    dim = group_size + 1
    p_00, p_01 = matrix0[0, 0], matrix0[0, 1]
    p_10, p_11 = matrix0[1, 0], matrix0[1, 1]
    T = np.zeros((dim, dim))
    for i in range(dim):
        s0, s1 = group_size - i, i
        for k in range(s0 + 1):
            for l in range(s1 + 1):
                # k = number of current-S0 agents that become S1;
                # l = number of current-S1 agents that remain S1.
                # The second binomial is therefore Binom(s1, p_11):
                # p_11**l * p_10**(s1-l).  Keeping these exponents in this
                # order is required for parity with the original 1-9 solver.
                prob = (_binom(s0, k) * (p_01 ** k) * (p_00 ** (s0 - k))
                        * _binom(s1, l) * (p_11 ** l) * (p_10 ** (s1 - l)))
                T[i, k + l] += prob
        s = T[i].sum()
        if s > 0:
            T[i] /= s
    return T


def precompute_transitions(group_sizes: List[int], matrix0: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    """One (N_k+1, N_k+1) idle transition matrix per group (K=2)."""
    if isinstance(matrix0, np.ndarray):
        matrix0 = [matrix0] * len(group_sizes)
    return [build_transition_matrix(matrix0[g], N) for g, N in enumerate(group_sizes)]


def _build_win_matrix(matrix0: np.ndarray, matrix1: np.ndarray,
                       group_size: int, winner_state: int) -> np.ndarray:
    """K=2 win transition matrix for one group."""
    N = group_size
    M = np.zeros((N + 1, N + 1))
    if N == 0:
        return M
    M_rest = build_transition_matrix(matrix0, max(0, N - 1))
    p_win_next = matrix1[winner_state, :]

    for i in range(N + 1):
        if winner_state == 0 and i == N:
            continue
        if winner_state == 1 and i == 0:
            continue

        if winner_state == 0:
            rest_s0, rest_s1 = N - i - 1, i
        else:
            rest_s0, rest_s1 = N - i, i - 1

        if rest_s0 < 0 or rest_s1 < 0:
            continue
        rest_idx = rest_s1

        for w in [0, 1]:
            pw = p_win_next[w]
            if pw == 0:
                continue
            for k, p_nw in enumerate(M_rest[rest_idx]):
                new_s1 = w + k
                if new_s1 <= N:
                    M[i, new_s1] += pw * p_nw

    for r in range(N + 1):
        s = M[r].sum()
        if s > 0:
            M[r] /= s
    return M


class _StateMapper:
    """Map compositional state tuples to a 1-D index."""

    def __init__(self, group_size: int, num_states: int):
        self.group_size = group_size
        self.num_states = num_states
        self.states: List[Tuple[int, ...]] = []
        self.state_to_idx: Dict[Tuple[int, ...], int] = {}
        for combo in itertools.product(*[range(group_size + 1)] * num_states):
            if sum(combo) == group_size:
                idx = len(self.states)
                self.states.append(combo)
                self.state_to_idx[combo] = idx
        self.num_states_total = len(self.states)

    def get_index(self, state: Tuple[int, ...]) -> int:
        return self.state_to_idx[state]

    def get_state(self, idx: int) -> Tuple[int, ...]:
        return self.states[idx]


class _TransitionMatrixBuilder:
    """Build idle/win transition matrices over compressed compositional states."""

    def __init__(self, group_size: int, matrix0: np.ndarray,
                  matrix1: np.ndarray, num_states: int):
        self.group_size = group_size
        self.matrix0 = matrix0
        self.matrix1 = matrix1
        self.num_states = num_states
        self.mapper = _StateMapper(group_size, num_states)
        self._zero_state = tuple([0] * num_states)

    def _group_transition_dict(self, current_state: Tuple[int, ...],
                                single_matrix: np.ndarray) -> Dict[Tuple[int, ...], float]:
        """Compute P(next_state | current_state) under independent agents."""
        K = self.num_states
        if sum(current_state) == 0:
            return {self._zero_state: 1.0}
        # Convolution over per-state-binomial draws
        final = {self._zero_state: 1.0}
        for s_idx, count in enumerate(current_state):
            if count == 0:
                continue
            probs = single_matrix[s_idx]
            # All compositions (x_0, ..., x_{K-1}) summing to count
            sub = {}
            for combo in itertools.product(*[range(count + 1)] * K):
                if sum(combo) != count:
                    continue
                p = 1.0
                for k in range(K):
                    p *= _binom(count, combo[k]) if k == 0 else (
                        _binom(count - sum(combo[:k]), combo[k]))
                # The _binom accumulation above is fragile; use multinomial directly
                p = 1.0
                rem = count
                for k in range(K - 1):
                    p *= _binom(rem, combo[k])
                    rem -= combo[k]
                if p * np.prod(probs ** np.array(combo)) > 1e-12:
                    sub[combo] = p * float(np.prod(np.array(probs) ** np.array(combo)))
            # Convolve
            new_dist: Dict[Tuple[int, ...], float] = {}
            for s1, p1 in final.items():
                for s2, p2 in sub.items():
                    combined = tuple(a + b for a, b in zip(s1, s2))
                    new_dist[combined] = new_dist.get(combined, 0.0) + p1 * p2
            final = new_dist
        return final

    def build_idle_matrix(self) -> np.ndarray:
        n = self.mapper.num_states_total
        T = np.zeros((n, n))
        for from_idx, from_state in enumerate(self.mapper.states):
            probs = self._group_transition_dict(from_state, self.matrix0)
            for to_state, prob in probs.items():
                T[from_idx, self.mapper.get_index(to_state)] = prob
        return T

    def build_win_matrix(self, winner_state: int) -> Optional[np.ndarray]:
        if self.group_size == 0:
            return None
        n = self.mapper.num_states_total
        T = np.zeros((n, n))
        any_valid = False
        for from_idx, from_state in enumerate(self.mapper.states):
            if from_state[winner_state] == 0:
                continue
            any_valid = True
            remaining = list(from_state)
            remaining[winner_state] -= 1
            rem_probs = self._group_transition_dict(tuple(remaining), self.matrix0)
            win_probs = self.matrix1[winner_state]
            for rem_state, rem_p in rem_probs.items():
                for wnext in range(self.num_states):
                    wp = win_probs[wnext]
                    if wp == 0:
                        continue
                    final = list(rem_state)
                    final[wnext] += 1
                    T[from_idx, self.mapper.get_index(tuple(final))] += rem_p * wp
        return T if any_valid else None


def precompute_general_transitions(
    group_sizes: List[int],
    matrix0: Union[np.ndarray, List[np.ndarray]],
    matrix1: Union[np.ndarray, List[np.ndarray]],
    num_states: int,
) -> List[Dict[str, object]]:
    """
    For each group, return {'idle': (n,n), 'win': [(n,n) per winner_state]}.
    Used by the K>=3 path.
    """
    if isinstance(matrix0, np.ndarray):
        matrix0 = [matrix0] * len(group_sizes)
    if isinstance(matrix1, np.ndarray):
        matrix1 = [matrix1] * len(group_sizes)
    out = []
    for g, N in enumerate(group_sizes):
        builder = _TransitionMatrixBuilder(N, matrix0[g], matrix1[g], num_states)
        out.append({
            "idle": builder.build_idle_matrix(),
            "win": [builder.build_win_matrix(s) for s in range(num_states)],
            "mapper": builder.mapper,
        })
    return out


# For parity with the baseline, use the "no-i" single-buyer VCG payment
# formula: remove one buyer from group g and look up the optimal action in
# that no-i economy.  This differs from group-level VCG.

def _no_i_economy_indices(group_sizes: List[int], g_miss: int) -> List[int]:
    """Return sizes in no-i economy where group g_miss has one fewer buyer."""
    return [N - 1 if k == g_miss else N for k, N in enumerate(group_sizes)]


def _build_no_i_transitions_K2(
    sizes_no: List[int],
    matrix0: List[np.ndarray],
    matrix1: List[np.ndarray],
):
    """Pre-build idle/win transitions for a K=2 no-i economy (any m)."""
    M_idle = precompute_transitions(sizes_no, matrix0)
    M_win_s0 = [
        _build_win_matrix(matrix0[g], matrix1[g], sizes_no[g], 0)
        if sizes_no[g] > 0 else np.zeros((1, 1))
        for g in range(len(sizes_no))
    ]
    M_win_s1 = [
        _build_win_matrix(matrix0[g], matrix1[g], sizes_no[g], 1)
        if sizes_no[g] > 0 else np.zeros((1, 1))
        for g in range(len(sizes_no))
    ]
    return M_idle, M_win_s0, M_win_s1


def _no_i_einsum_future(mats, W_next, delta):
    """Einsum over m transition matrices applied to W_next, scaled by delta."""
    m = len(mats)
    rhs_chars = "abcdefghijklmnop"[:m]
    lhs_chars = "zyxwvutsrqponm"[:m]
    full_lhs = ",".join(f"{lhs_char}{rhs_char}"
                        for lhs_char, rhs_char in zip(lhs_chars, rhs_chars))
    out_chars = "".join(lhs_chars) + "g"
    rhs_full = "".join(rhs_chars) + "g"
    operands = mats + [W_next]
    expr = f"{full_lhs},{rhs_full}->{out_chars}"
    return np.einsum(expr, *operands) * delta


def _no_i_backward_induction_K2_m2(
    T: int,
    delta: float,
    valuations: np.ndarray,
    M_idle: List[np.ndarray],
    M_win_s0: List[np.ndarray],
    M_win_s1: List[np.ndarray],
    sizes_no: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No-i DP for K=2, m=2. Returns (policy, W, idle_welfare)."""
    N0p, N1p = sizes_no[0], sizes_no[1]
    shape = (N0p + 1, N1p + 1)
    W = np.zeros((T + 1,) + shape, dtype=np.float64)
    policy = np.full((T,) + shape, -1, dtype=np.int8)
    idle_welfare = np.zeros((T,) + shape, dtype=np.float64)

    for t in range(T - 1, -1, -1):
        W_next = W[t + 1]
        T0_idle, T1_idle = M_idle[0], M_idle[1]
        fv_idle = delta * (T0_idle @ W_next @ T1_idle.T)
        W_curr = fv_idle.copy()
        pol_curr = np.full(shape, -1, dtype=np.int8)

        if sizes_no[0] > 0:
            T0_ws0, T0_ws1 = M_win_s0[0], M_win_s1[0]
            cur = delta * (T0_ws0 @ W_next @ T1_idle.T) + float(valuations[0, 0])
            mk = (np.arange(N0p + 1) < N0p)[:, None]
            upd = mk & (cur > W_curr)
            W_curr[upd] = cur[upd]
            pol_curr[upd] = 0

            cur = delta * (T0_ws1 @ W_next @ T1_idle.T) + float(valuations[0, 1])
            mk = (np.arange(N0p + 1) > 0)[:, None]
            upd = mk & (cur > W_curr)
            W_curr[upd] = cur[upd]
            pol_curr[upd] = 1

        if sizes_no[1] > 0:
            T1_ws0, T1_ws1 = M_win_s0[1], M_win_s1[1]
            cur = delta * (T0_idle @ W_next @ T1_ws0.T) + float(valuations[1, 0])
            mk = (np.arange(N1p + 1) < N1p)[None, :]
            upd = mk & (cur > W_curr)
            W_curr[upd] = cur[upd]
            pol_curr[upd] = 2

            cur = delta * (T0_idle @ W_next @ T1_ws1.T) + float(valuations[1, 1])
            mk = (np.arange(N1p + 1) > 0)[None, :]
            upd = mk & (cur > W_curr)
            W_curr[upd] = cur[upd]
            pol_curr[upd] = 3

        W[t] = W_curr
        policy[t] = pol_curr
        idle_welfare[t] = fv_idle
    return policy, W, idle_welfare


def _no_i_backward_induction_K2_mg(
    T: int,
    delta: float,
    valuations: np.ndarray,
    M_idle: List[np.ndarray],
    M_win_s0: List[np.ndarray],
    M_win_s1: List[np.ndarray],
    sizes_no: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No-i DP for K=2, m in {3,4}. Returns (policy, W, idle_welfare)."""
    m = len(sizes_no)
    shape_no = tuple(N + 1 for N in sizes_no)
    W = np.zeros((T + 1,) + shape_no + (m,), dtype=np.float64)
    policy = np.full((T,) + shape_no, -1, dtype=np.int8)
    idle_welfare = np.zeros((T,) + shape_no + (m,), dtype=np.float64)

    for t in range(T - 1, -1, -1):
        W_next = W[t + 1]
        fv_idle = _no_i_einsum_future(M_idle, W_next, delta)
        W_curr = fv_idle.copy()
        pol_curr = np.full(shape_no, -1, dtype=np.int8)

        for win_g in range(m):
            if sizes_no[win_g] == 0:
                continue
            mats = list(M_idle)
            # Win via S0
            mats[win_g] = M_win_s0[win_g]
            fv_s0 = _no_i_einsum_future(mats, W_next, delta)
            curr_s0 = fv_s0.copy()
            curr_s0[..., win_g] += float(valuations[win_g, 0])
            mask_s0 = _axis_mask(shape_no, win_g, "<", sizes_no[win_g])
            sum_s0 = curr_s0.sum(axis=-1)
            best = W_curr
            best_sum = best.sum(axis=-1)
            improve = (sum_s0 > best_sum) & mask_s0
            best[improve] = curr_s0[improve]
            best_sum[improve] = sum_s0[improve]
            pol_curr[improve] = 2 * win_g  # 2*g + 0

            # Win via S1
            mats[win_g] = M_win_s1[win_g]
            fv_s1 = _no_i_einsum_future(mats, W_next, delta)
            curr_s1 = fv_s1.copy()
            curr_s1[..., win_g] += float(valuations[win_g, 1])
            mask_s1 = _axis_mask(shape_no, win_g, ">", 0)
            sum_s1 = curr_s1.sum(axis=-1)
            improve = (sum_s1 > best_sum) & mask_s1
            best[improve] = curr_s1[improve]
            best_sum[improve] = sum_s1[improve]
            pol_curr[improve] = 2 * win_g + 1  # 2*g + 1

        W[t] = W_curr
        policy[t] = pol_curr
        idle_welfare[t] = fv_idle
    return policy, W, idle_welfare


def _no_i_get_exp_val_K2_m2(
    ac: int,
    i_no: int,
    j_no: int,
    W_next: np.ndarray,
    M_idle: List[np.ndarray],
    M_win_s0: List[np.ndarray],
    M_win_s1: List[np.ndarray],
) -> float:
    """Expected value for action ac in the K=2, m=2 no-i economy."""
    T0_idle, T1_idle = M_idle[0], M_idle[1]
    if ac == -1:
        v0 = M_idle[0][i_no, :]
        v1 = M_idle[1][j_no, :]
    elif ac == 0:
        v0 = M_win_s0[0][i_no, :]
        v1 = M_idle[1][j_no, :]
    elif ac == 1:
        v0 = M_win_s1[0][i_no, :]
        v1 = M_idle[1][j_no, :]
    elif ac == 2:
        v0 = M_idle[0][i_no, :]
        v1 = M_win_s0[1][j_no, :]
    elif ac == 3:
        v0 = M_idle[0][i_no, :]
        v1 = M_win_s1[1][j_no, :]
    else:
        raise ValueError(f"unknown action code: {ac}")
    return float(v0 @ W_next @ v1.T)


def _no_i_payment_K2_m2(
    T: int,
    delta: float,
    valuations: np.ndarray,
    M_idle_no: List[np.ndarray],
    M_win_s0_no: List[np.ndarray],
    M_win_s1_no: List[np.ndarray],
    policy_no: np.ndarray,
    W_no: np.ndarray,
    g_miss: int,
    full_policy: np.ndarray,
    N0: int,
    N1: int,
) -> np.ndarray:
    """Apply the no-i single-buyer VCG payment formula.

    Returns pay_grid of shape (T, N0+1, N1+1) for group g_miss.
    """
    pay_grid = np.zeros((T, N0 + 1, N1 + 1), dtype=np.float64)
    for t in range(T):
        W_no_next = W_no[t + 1]
        for i in range(N0 + 1):
            for j in range(N1 + 1):
                ac = int(full_policy[t, i, j])
                if ac == -1:
                    pay_grid[t, i, j] = 0.0
                    continue
                act_g = 0 if ac < 2 else 1
                act_s = ac % 2
                # Compute (i_no, j_no) per 1-9 formula.
                if g_miss == 0 and act_s == 1:
                    i_no = i - 1
                    j_no = j
                elif g_miss == 1 and act_s == 1:
                    i_no = i
                    j_no = j - 1
                else:
                    i_no = i
                    j_no = j
                # Out-of-range means the action is infeasible in no-i (e.g.,
                # when the winner's group has no remaining members).  This
                # should not happen for valid actions; fall back to 0.
                if i_no < 0 or j_no < 0:
                    pay_grid[t, i, j] = 0.0
                    continue
                if g_miss == 0 and i_no >= M_idle_no[0].shape[0]:
                    pay_grid[t, i, j] = 0.0
                    continue
                if g_miss == 1 and j_no >= M_idle_no[1].shape[0]:
                    pay_grid[t, i, j] = 0.0
                    continue
                # term1: look up no-i policy's optimal action and its valuation
                ac_no = int(policy_no[t, i_no, j_no])
                term1 = 0.0
                if ac_no != -1:
                    act_g_no = 0 if ac_no < 2 else 1
                    act_s_no = ac_no % 2
                    term1 = float(valuations[act_g_no, act_s_no])
                # term2: delta * (W_no[w*] - W_no[idle]) evaluated at (i_no, j_no)
                w_optimal = _no_i_get_exp_val_K2_m2(
                    ac_no, i_no, j_no, W_no_next,
                    M_idle_no, M_win_s0_no, M_win_s1_no)
                w_actual = _no_i_get_exp_val_K2_m2(
                    -1, i_no, j_no, W_no_next,
                    M_idle_no, M_win_s0_no, M_win_s1_no)
                term2 = delta * (w_optimal - w_actual)
                pay_grid[t, i, j] = term1 + term2
    return pay_grid



def solve_analytical_p0_vectorized(
    w_g0_if_g0_wins: np.ndarray,
    w_g0_if_g1_wins: np.ndarray,
    w_g1_if_g0_wins: np.ndarray,
    w_g1_if_g1_wins: np.ndarray,
    group_sizes: tuple,
    epsilon: float
) -> np.ndarray:
    """Closed-form solution for the m=2 equity-constrained optimization."""
    N0, N1 = group_sizes
    N0 = max(N0, 1e-9)
    N1 = max(N1, 1e-9)

    A0 = w_g0_if_g0_wins - w_g0_if_g1_wins
    B0 = w_g0_if_g1_wins

    A1 = w_g1_if_g0_wins - w_g1_if_g1_wins
    B1 = w_g1_if_g1_wins

    obj_slope = A0 + A1

    C = A0 / N0 - A1 / N1
    D = B0 / N0 - B1 / N1

    lower_bound = -epsilon - D
    upper_bound = epsilon - D

    p_min = np.zeros_like(C)
    p_max = np.ones_like(C)

    nonzero_mask = np.abs(C) > 1e-12

    val1 = np.zeros_like(C)
    val2 = np.zeros_like(C)

    np.divide(lower_bound, C, out=val1, where=nonzero_mask)
    np.divide(upper_bound, C, out=val2, where=nonzero_mask)

    c_pos = (C > 1e-12)
    p_min = np.where(c_pos, np.maximum(p_min, val1), p_min)
    p_max = np.where(c_pos, np.minimum(p_max, val2), p_max)

    c_neg = (C < -1e-12)
    p_min = np.where(c_neg, np.maximum(p_min, val2), p_min)
    p_max = np.where(c_neg, np.minimum(p_max, val1), p_max)

    feasible = p_min <= p_max + 1e-9

    final_p = np.clip((p_min + p_max) / 2, 0, 1)

    slope_pos = (obj_slope > 0) & feasible
    final_p = np.where(slope_pos, p_max, final_p)

    slope_neg = (obj_slope < 0) & feasible
    final_p = np.where(slope_neg, p_min, final_p)

    return np.clip(final_p, 0.0, 1.0)



def solve_optimization_mg_lp(
    welfare_scenarios: np.ndarray,    # (batch, m, m)  rows = per-group welfare under each scenario
    group_sizes: List[int],
    epsilon: float,
) -> np.ndarray:
    """Solve the fairness-constrained LP for each row of welfare_scenarios.

    welfare_scenarios has shape (batch, m, m): welfare of group g when
    scenario s wins. Returns p on the probability simplex.
    Falls back to equal weights on failure.
    """
    m = len(group_sizes)
    if m < 3:
        # Should not be called for m=2; safety net
        return np.full((welfare_scenarios.shape[0], m), 1.0 / m)

    batch = welfare_scenarios.shape[0]
    results = np.zeros((batch, m), dtype=np.float64)

    Ns = np.array([max(float(n), 1e-9) for n in group_sizes], dtype=np.float64)

    if not _HAS_LINPROG:
        # Equal split fallback
        results[:] = 1.0 / m
        return results

    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]

    # Objective: -sum_g V_g (linprog minimizes)
    c_obj = -np.sum(welfare_scenarios, axis=1)        # (batch, m)
    # Equality: sum(p) = 1
    A_eq = np.ones((1, m), dtype=np.float64)
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * m

    for i in range(batch):
        try:
            V = welfare_scenarios[i]                   # (m, m)
            V_norm = V / Ns[:, None]                   # (m, m), per-capita
            A_ub_rows = []
            b_ub_rows = []
            for g_i, g_j in pairs:
                D = V_norm[g_i] - V_norm[g_j]
                A_ub_rows.append(D.tolist())
                b_ub_rows.append(float(epsilon))
                A_ub_rows.append((-D).tolist())
                b_ub_rows.append(float(epsilon))
            A_ub = np.array(A_ub_rows, dtype=np.float64)
            b_ub = np.array(b_ub_rows, dtype=np.float64)

            res = linprog(
                c=c_obj[i], A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            if res.success and np.all(np.isfinite(res.x)):
                p = np.clip(res.x, 0.0, 1.0)
                s = p.sum()
                results[i] = p / s if s > 0 else np.full(m, 1.0 / m)
            else:
                results[i] = np.full(m, 1.0 / m)
        except Exception:
            results[i] = np.full(m, 1.0 / m)

    return results


def solve_optimization_3g_vectorized(
    w_g0_scenarios, w_g1_scenarios, w_g2_scenarios,
    group_sizes: tuple, epsilon: float,
) -> np.ndarray:
    """3-group drop-in replacement for the scalar API."""
    arrs0 = [np.asarray(x, dtype=np.float64).reshape(-1) for x in w_g0_scenarios]
    arrs1 = [np.asarray(x, dtype=np.float64).reshape(-1) for x in w_g1_scenarios]
    arrs2 = [np.asarray(x, dtype=np.float64).reshape(-1) for x in w_g2_scenarios]
    N = arrs0[0].size
    welfare = np.zeros((N, 3, 3), dtype=np.float64)
    for s in range(3):
        welfare[:, 0, s] = arrs0[s]
        welfare[:, 1, s] = arrs1[s]
        welfare[:, 2, s] = arrs2[s]
    return solve_optimization_m3_lp(welfare, list(group_sizes), epsilon)


def solve_optimization_4g_vectorized(
    w_g0_scenarios, w_g1_scenarios, w_g2_scenarios, w_g3_scenarios,
    group_sizes: tuple, epsilon: float,
) -> np.ndarray:
    """4-group drop-in replacement for the scalar API."""
    arrs = [
        [np.asarray(x, dtype=np.float64).reshape(-1) for x in w_g0_scenarios],
        [np.asarray(x, dtype=np.float64).reshape(-1) for x in w_g1_scenarios],
        [np.asarray(x, dtype=np.float64).reshape(-1) for x in w_g2_scenarios],
        [np.asarray(x, dtype=np.float64).reshape(-1) for x in w_g3_scenarios],
    ]
    N = arrs[0][0].size
    welfare = np.zeros((N, 4, 4), dtype=np.float64)
    for s in range(4):
        for g in range(4):
            welfare[:, g, s] = arrs[g][s]
    return solve_optimization_mg_lp(welfare, list(group_sizes), epsilon)


class M3SolverTimeout(RuntimeError):
    """Raised when the m=3 analytical LP solver exceeds its budget."""


def _solve_m3_via_subprocess(
    welfare_scenarios: np.ndarray,
    group_sizes: List[int],
    epsilon: float,
    timeout_sec: Optional[float] = 60.0,
) -> np.ndarray:
    """Call the standalone m=3 LP solver in a subprocess with a wall-clock timeout."""
    import subprocess, pickle, os, sys
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_m3_solver_worker.py")
    data = {
        "welfare_scenarios": welfare_scenarios,
        "group_sizes": group_sizes,
        "epsilon": float(epsilon),
        "timeout_sec": float(timeout_sec) if timeout_sec is not None else 0.0,
    }
    proc = subprocess.Popen(
        [sys.executable, script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = proc.communicate(pickle.dumps(data), timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise M3SolverTimeout(
            f"m3 solver exceeded {timeout_sec}s wall-clock budget"
        )
    # Worker uses sys.exit(73) to signal CPU-budget expiry.
    if proc.returncode == 73:
        raise M3SolverTimeout(
            f"m3 solver CPU-budget exceeded (> {data['timeout_sec']}s budget)"
        )
    if proc.returncode != 0:
        raise RuntimeError(f"m3 solver failed: {err.decode(errors='ignore')[:500]}")
    result = pickle.loads(out)
    return result["p"].astype(np.float64)


def solve_optimization_m3_lp(
    welfare_scenarios: np.ndarray,
    group_sizes: List[int],
    epsilon: float,
) -> np.ndarray:
    """
    Analytical vertex-enumeration LP solver for m=3, K=2.
    Avoids scipy linprog, which can segfault on large grids.
    """
    N = welfare_scenarios.shape[0]
    N0, N1, N2 = group_sizes
    N0 = max(N0, 1e-9)
    N1 = max(N1, 1e-9)
    N2 = max(N2, 1e-9)

    results = np.zeros((N, 3), dtype=np.float64)

    c0 = -(welfare_scenarios[:, 0, 0] + welfare_scenarios[:, 1, 0] + welfare_scenarios[:, 2, 0])
    c1 = -(welfare_scenarios[:, 0, 1] + welfare_scenarios[:, 1, 1] + welfare_scenarios[:, 2, 1])
    c2 = -(welfare_scenarios[:, 0, 2] + welfare_scenarios[:, 1, 2] + welfare_scenarios[:, 2, 2])

    V_norm = np.zeros_like(welfare_scenarios, dtype=np.float64)
    V_norm[:, 0, :] = welfare_scenarios[:, 0, :] / N0
    V_norm[:, 1, :] = welfare_scenarios[:, 1, :] / N1
    V_norm[:, 2, :] = welfare_scenarios[:, 2, :] / N2

    pairs = [(0, 1), (0, 2), (1, 2)]
    default = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    for i in range(N):
        obj_a = c0[i] - c2[i]
        obj_b = c1[i] - c2[i]

        constraints = [
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (-1.0, -1.0, 0.0),
        ]
        for g_i, g_j in pairs:
            a = (V_norm[i, g_i, 0] - V_norm[i, g_i, 2]) - (V_norm[i, g_j, 0] - V_norm[i, g_j, 2])
            b = (V_norm[i, g_i, 1] - V_norm[i, g_i, 2]) - (V_norm[i, g_j, 1] - V_norm[i, g_j, 2])
            const = V_norm[i, g_i, 2] - V_norm[i, g_j, 2]
            constraints.append((a, b, float(epsilon) - const))
            constraints.append((-a, -b, float(epsilon) + const))
        constraints = np.array(constraints, dtype=np.float64).copy()
        n = len(constraints)

        best_val = -np.inf
        best_p = default.copy()
        feasible_found = False

        for ci in range(n):
            for cj in range(ci + 1, n):
                a1, b1, rhs1 = constraints[ci]
                a2, b2, rhs2 = constraints[cj]
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-12:
                    continue
                p0 = (rhs1 * b2 - rhs2 * b1) / det
                p1 = (a1 * rhs2 - a2 * rhs1) / det
                p2 = 1.0 - p0 - p1
                if p0 < -1e-9 or p1 < -1e-9 or p2 < -1e-9:
                    continue
                if p0 > 1.0 + 1e-9 or p1 > 1.0 + 1e-9 or p2 > 1.0 + 1e-9:
                    continue
                ok = True
                for k in range(n):
                    if constraints[k, 0] * p0 + constraints[k, 1] * p1 > constraints[k, 2] + 1e-9:
                        ok = False
                        break
                if not ok:
                    continue
                feasible_found = True
                val = obj_a * p0 + obj_b * p1
                if val > best_val + 1e-12:
                    best_val = val
                    best_p = np.array([p0, p1, p2], dtype=np.float64)

        if feasible_found:
            best_p = np.clip(best_p, 0.0, 1.0)
            s = best_p.sum()
            best_p = best_p / s if s > 0 else default
        results[i] = best_p

    return results



def solve_eq1_one(
    sw: np.ndarray,
    sw_remaining: np.ndarray,
    group_sizes: List[int],
    epsilon: float,
) -> np.ndarray:
    """Solve the equity-constrained optimization for a single (s,t), m >= 2."""
    total = sw + sw_remaining

    m = len(group_sizes)
    if m == 2:
        N0, N1 = group_sizes[0], group_sizes[1]
        p0 = solve_analytical_p0_vectorized(
            np.array([total[0]]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([total[1]]),
            (N0, N1),
            epsilon,
        )
        p0 = float(p0[0])
        return np.array([p0, 1.0 - p0], dtype=np.float64)

    per_cap = total / np.asarray(group_sizes, dtype=np.float64)
    order = np.argsort(-per_cap)
    selected = np.ones(m, dtype=bool)
    while True:
        pc = per_cap[selected]
        if pc.max() - pc.min() <= epsilon + 1e-9:
            break
        idx_drop = np.where(selected)[0][np.argmin(per_cap[selected])]
        selected[idx_drop] = False
        if selected.sum() == 0:
            selected = np.zeros(m, dtype=bool)
            selected[np.argmax(per_cap)] = True
            break

    sel_idx = np.where(selected)[0]
    phi = np.zeros(m, dtype=np.float64)
    if len(sel_idx) > 0:
        w = total[sel_idx]
        w = np.maximum(w, 1e-12)
        phi[sel_idx] = w / w.sum()
    else:
        phi[np.argmax(total)] = 1.0
    return phi



def fair_pivot_bi(
    group_sizes: List[int],
    num_states: int,
    T: int,
    delta: float,
    epsilon: float,
    valuations: np.ndarray,                         # (m, K)
    matrix0: Union[np.ndarray, List[np.ndarray]],   # (K, K) or list of m
    matrix1: Union[np.ndarray, List[np.ndarray]],   # (K, K) or list of m
    compute_payment: bool = True,
) -> Dict:
    """Backward induction oracle dispatcher.

    Returns dict with keys: states, phi, V, and payment (if requested).
    """
    m = len(group_sizes)
    if num_states == 2 and m == 2:
        return _fair_pivot_bi_K2_m2(
            group_sizes, T, delta, epsilon, valuations, matrix0, matrix1,
            compute_payment=compute_payment)
    if num_states == 2 and m in (3, 4):
        return _fair_pivot_bi_K2_mg(
            group_sizes, T, delta, epsilon, valuations, matrix0, matrix1,
            compute_payment=compute_payment)
    if num_states in (3, 4) and m == 2:
        return _fair_pivot_bi_Kg_m2(
            group_sizes, num_states, T, delta, epsilon, valuations, matrix0,
            matrix1, compute_payment=compute_payment)
    # K>=3, m>=3 — not needed by current experiments but supported via
    # a generalized grid solver.
    if num_states in (3, 4) and m in (3, 4):
        return _fair_pivot_bi_Kg_mg(
            group_sizes, num_states, T, delta, epsilon, valuations, matrix0,
            matrix1, compute_payment=compute_payment)
    raise NotImplementedError(
        f"No oracle path for m={m}, K={num_states}. "
        f"Supported: m in {{2,3,4}}, K in {{2,3,4}}.")



def _fair_pivot_bi_K2_m2(
    group_sizes: List[int],
    T: int,
    delta: float,
    epsilon: float,
    valuations: np.ndarray,
    matrix0: Union[np.ndarray, List[np.ndarray]],
    matrix1: Union[np.ndarray, List[np.ndarray]],
    compute_payment: bool = True,
) -> Dict:
    """K=2, m=2 closed-form LP solver."""
    if isinstance(matrix0, np.ndarray):
        matrix0 = [matrix0] * len(group_sizes)
    if isinstance(matrix1, np.ndarray):
        matrix1 = [matrix1] * len(group_sizes)
    assert num_states_check(matrix0[0], 2), "K=2 path expects 2x2 matrix"
    m = len(group_sizes)
    assert m == 2, f"K=2/m=2 path called with m={m}"

    states = enumerate_market_states(group_sizes, 2)
    S = states.shape[0]

    M_idle = precompute_transitions(group_sizes, matrix0)
    M_win_s0 = []
    M_win_s1 = []
    for k in range(m):
        Nk = group_sizes[k]
        M_win_s0.append(_build_win_matrix(matrix0[k], matrix1[k], Nk, 0) if Nk > 0 else np.zeros((1, 1)))
        M_win_s1.append(_build_win_matrix(matrix0[k], matrix1[k], Nk, 1) if Nk > 0 else np.zeros((1, 1)))

    shape_per_group = tuple(Nk + 1 for Nk in group_sizes)
    SW = np.zeros((T + 1,) + shape_per_group + (m,), dtype=np.float64)
    optimal_p0 = np.zeros((T,) + shape_per_group, dtype=np.float64)
    # Action code per (t, i, j): -1 idle, 0 G0-S0, 1 G0-S1, 2 G1-S0, 3 G1-S1.
    # Needed by the no-i VCG payment formula (1-9 parity).
    optimal_action = np.full((T,) + shape_per_group, -1, dtype=np.int8)

    v_k_s0 = valuations[:, 0].copy()
    v_k_s1 = valuations[:, 1].copy()

    for t in range(T - 1, -1, -1):
        SW_next = SW[t + 1]

        N0, N1 = group_sizes[0], group_sizes[1]

        def _exp_future(T0, T1):
            nw_0 = SW_next[..., 0]
            nw_1 = SW_next[..., 1]
            exp_0 = T0 @ nw_0 @ T1.T
            exp_1 = T0 @ nw_1 @ T1.T
            return exp_0 * delta, exp_1 * delta

        T0_idle, T1_idle = M_idle[0], M_idle[1]

        fv_idle_0, fv_idle_1 = _exp_future(T0_idle, T1_idle)

        w_g0_if_g0_wins = fv_idle_0.copy()
        w_g1_if_g0_wins = fv_idle_1.copy()
        a_g0_wins = np.full((N0 + 1, N1 + 1), -1, dtype=np.int8)

        if N0 > 0:
            T0_ws0, T0_ws1 = M_win_s0[0], M_win_s1[0]
            fv0, fv1 = _exp_future(T0_ws0, T1_idle)
            curr0 = fv0 + v_k_s0[0]
            mask_s0 = np.arange(N0 + 1) < N0
            update = (curr0 > w_g0_if_g0_wins) & mask_s0[:, None]
            w_g0_if_g0_wins[update] = curr0[update]
            w_g1_if_g0_wins[update] = fv1[update]
            a_g0_wins[update] = 0

            fv0, fv1 = _exp_future(T0_ws1, T1_idle)
            curr1 = fv0 + v_k_s1[0]
            mask_s1 = np.arange(N0 + 1) > 0
            update = (curr1 > w_g0_if_g0_wins) & mask_s1[:, None]
            w_g0_if_g0_wins[update] = curr1[update]
            w_g1_if_g0_wins[update] = fv1[update]
            a_g0_wins[update] = 1

        w_g0_if_g1_wins = fv_idle_0.copy()
        w_g1_if_g1_wins = fv_idle_1.copy()
        a_g1_wins = np.full((N0 + 1, N1 + 1), -1, dtype=np.int8)

        if N1 > 0:
            T1_ws0, T1_ws1 = M_win_s0[1], M_win_s1[1]
            fv0, fv1 = _exp_future(T0_idle, T1_ws0)
            curr0 = fv1 + v_k_s0[1]
            mask_s0 = np.arange(N1 + 1) < N1
            update = (curr0 > w_g1_if_g1_wins) & mask_s0[None, :]
            w_g1_if_g1_wins[update] = curr0[update]
            w_g0_if_g1_wins[update] = fv0[update]
            a_g1_wins[update] = 2

            fv0, fv1 = _exp_future(T0_idle, T1_ws1)
            curr1 = fv1 + v_k_s1[1]
            mask_s1 = np.arange(N1 + 1) > 0
            update = (curr1 > w_g1_if_g1_wins) & mask_s1[None, :]
            w_g1_if_g1_wins[update] = curr1[update]
            w_g0_if_g1_wins[update] = fv0[update]
            a_g1_wins[update] = 3

        p0_grid = np.full((N0 + 1, N1 + 1), 0.5, dtype=np.float64)
        valid = np.ones_like(p0_grid, dtype=bool)

        if N0 == 0:
            p0_grid[:] = 0.0
            valid[:] = False
        elif N1 == 0:
            p0_grid[:] = 1.0
            valid[:] = False

        if np.any(valid):
            opt = solve_analytical_p0_vectorized(
                w_g0_if_g0_wins[valid],
                w_g0_if_g1_wins[valid],
                w_g1_if_g0_wins[valid],
                w_g1_if_g1_wins[valid],
                (N0, N1),
                epsilon,
            )
            p0_grid[valid] = opt

        p1_grid = 1.0 - p0_grid
        SW[t, ..., 0] = p0_grid * w_g0_if_g0_wins + p1_grid * w_g0_if_g1_wins
        SW[t, ..., 1] = p0_grid * w_g1_if_g0_wins + p1_grid * w_g1_if_g1_wins
        optimal_p0[t] = p0_grid
        # Record the full mechanism's action code using the higher-welfare scenario.
        sum_g0_scenario = w_g0_if_g0_wins + w_g1_if_g0_wins
        sum_g1_scenario = w_g0_if_g1_wins + w_g1_if_g1_wins
        choose_g0 = sum_g0_scenario >= sum_g1_scenario
        ac_grid = np.where(choose_g0, a_g0_wins, a_g1_wins)
        # If both candidate actions are -1, idle.
        both_idle = (a_g0_wins == -1) & (a_g1_wins == -1)
        ac_grid = np.where(both_idle, np.int8(-1), ac_grid)
        optimal_action[t] = ac_grid

    if not compute_payment:
        phi_star = np.zeros((S, T, m), dtype=np.float64)
        V_star = np.zeros((S, T), dtype=np.float64)
        for s_idx in range(S):
            s_flat = states[s_idx]
            idx = (int(s_flat[1]), int(s_flat[3]))
            for t in range(T):
                p0 = float(optimal_p0[t][idx])
                phi_star[s_idx, t, 0] = p0
                phi_star[s_idx, t, 1] = 1.0 - p0
                V_star[s_idx, t] = float(SW[t][idx + (0,)] + SW[t][idx + (1,)])
        return _pack_result(states, phi_star, V_star, None, group_sizes,
                            num_states=2, T=T, delta=delta, epsilon=epsilon,
                            valuations=valuations, matrix0=matrix0, matrix1=matrix1)

    # No-i single-buyer VCG payments (baseline parity).
    pay_g0_grid = np.zeros((T,) + shape_per_group, dtype=np.float64)
    pay_g1_grid = np.zeros((T,) + shape_per_group, dtype=np.float64)

    for g_miss, pay_grid in [(0, pay_g0_grid), (1, pay_g1_grid)]:
        # Skip when group g_miss has no buyer; payment is trivially 0.
        if group_sizes[g_miss] == 0:
            continue
        sizes_no = _no_i_economy_indices(group_sizes, g_miss)
        M_idle_no, M_win_s0_no, M_win_s1_no = _build_no_i_transitions_K2(
            sizes_no, matrix0, matrix1)
        policy_no, W_no, _idle = _no_i_backward_induction_K2_m2(
            T, delta, valuations,
            M_idle_no, M_win_s0_no, M_win_s1_no, sizes_no)
        # Apply the no-i payment formula at every (t, i, j).
        pay_full = _no_i_payment_K2_m2(
            T, delta, valuations,
            M_idle_no, M_win_s0_no, M_win_s1_no,
            policy_no, W_no,
            g_miss=g_miss,
            full_policy=optimal_action,
            N0=group_sizes[0], N1=group_sizes[1])
        # Zero out payment unless the winner is g_miss.
        winner_g = np.where(optimal_action < 0, -1,
                            np.where(optimal_action < 2, 0, 1))
        mask = (winner_g == g_miss)
        pay_grid[:] = np.where(mask, pay_full, 0.0)

    phi_star = np.zeros((S, T, m), dtype=np.float64)
    V_star = np.zeros((S, T), dtype=np.float64)
    p_star = np.zeros((S, T, m), dtype=np.float64)

    for s_idx in range(S):
        s_flat = states[s_idx]
        idx = (int(s_flat[1]), int(s_flat[3]))
        for t in range(T):
            p0 = float(optimal_p0[t][idx])
            p1 = 1.0 - p0
            phi_star[s_idx, t, 0] = p0
            phi_star[s_idx, t, 1] = p1
            V_star[s_idx, t] = float(SW[t][idx + (0,)] + SW[t][idx + (1,)])
            # Expected payment is phi_g times the conditional pay_g.
            p_star[s_idx, t, 0] = p0 * float(pay_g0_grid[t][idx])
            p_star[s_idx, t, 1] = p1 * float(pay_g1_grid[t][idx])

    return _pack_result(states, phi_star, V_star, p_star, group_sizes,
                         num_states=2, T=T, delta=delta, epsilon=epsilon,
                         valuations=valuations, matrix0=matrix0, matrix1=matrix1,
                         optimal_action_full=optimal_action)



def _fair_pivot_bi_K2_mg(
    group_sizes: List[int],
    T: int,
    delta: float,
    epsilon: float,
    valuations: np.ndarray,
    matrix0: Union[np.ndarray, List[np.ndarray]],
    matrix1: Union[np.ndarray, List[np.ndarray]],
    compute_payment: bool = True,
) -> Dict:
    """K=2, m in {3,4}: full grid + LP via linprog."""
    if isinstance(matrix0, np.ndarray):
        matrix0 = [matrix0] * len(group_sizes)
    if isinstance(matrix1, np.ndarray):
        matrix1 = [matrix1] * len(group_sizes)
    m = len(group_sizes)
    assert m in (3, 4), f"K=2/mg path called with m={m}"

    states = enumerate_market_states(group_sizes, 2)
    S = states.shape[0]

    # Per-group transition matrices
    M_idle = precompute_transitions(group_sizes, matrix0)
    M_win_s0 = [_build_win_matrix(matrix0[g], matrix1[g], N, 0) for g, N in enumerate(group_sizes)]
    M_win_s1 = [_build_win_matrix(matrix0[g], matrix1[g], N, 1) for g, N in enumerate(group_sizes)]

    shape_per_group = tuple(Nk + 1 for Nk in group_sizes)
    SW = np.zeros((T + 1,) + shape_per_group + (m,), dtype=np.float64)
    optimal_p = np.zeros((T,) + shape_per_group + (m,), dtype=np.float64)
    # Action code per (t, idx_no_g): -1 idle, 2k (Gk-S0 win), 2k+1 (Gk-S1 win).
    # Used by the no-i VCG payment formula (1-9 parity).
    optimal_action_full = np.full((T,) + shape_per_group, -1, dtype=np.int8)

    v_k_s0 = valuations[:, 0].copy()
    v_k_s1 = valuations[:, 1].copy()

    # Build einsum string for m groups: 'il,jm,kn,...,lmn...g -> ijk...g'
    def _exp_future_einsum(mats):
        # mats: list of m matrices (N_k+1, N_k+1)
        rhs_chars = "abcdefghijklmnop"[:m]                # l,m,n,...
        lhs_chars = "zyxwvutsrqponm"[:m]                  # i,j,k,...
        full_lhs = ",".join(f"{lhs_char}{rhs_char}"
                            for lhs_char, rhs_char in zip(lhs_chars, rhs_chars))
        out_chars = "".join(lhs_chars) + "g"
        rhs_full = "".join(rhs_chars) + "g"
        operands = mats + [SW_next]
        expr = f"{full_lhs},{rhs_full}->{out_chars}"
        return np.einsum(expr, *operands) * delta

    for t in range(T - 1, -1, -1):
        SW_next = SW[t + 1]

        # Idle future
        fv_idle = _exp_future_einsum(M_idle)               # shape_per_group + (m,)

        # Each "group g wins" scenario, starting from idle baseline
        welfare_scenarios = [fv_idle.copy() for _ in range(m)]

        for win_g in range(m):
            if group_sizes[win_g] == 0:
                continue
            mats = list(M_idle)

            # Win via S0
            mats[win_g] = M_win_s0[win_g]
            fv_s0 = _exp_future_einsum(mats)
            curr_s0 = fv_s0.copy()
            curr_s0[..., win_g] += v_k_s0[win_g]
            mask_s0 = _axis_mask(shape_per_group, win_g, "<", group_sizes[win_g])
            sum_s0 = curr_s0.sum(axis=-1)
            best = welfare_scenarios[win_g]
            best_sum = best.sum(axis=-1)
            improve = (sum_s0 > best_sum) & mask_s0
            best[improve] = curr_s0[improve]
            best_sum[improve] = sum_s0[improve]
            optimal_action_full[t][improve] = 2 * win_g  # Gk-S0

            # Win via S1
            mats[win_g] = M_win_s1[win_g]
            fv_s1 = _exp_future_einsum(mats)
            curr_s1 = fv_s1.copy()
            curr_s1[..., win_g] += v_k_s1[win_g]
            mask_s1 = _axis_mask(shape_per_group, win_g, ">", 0)
            sum_s1 = curr_s1.sum(axis=-1)
            improve = (sum_s1 > best_sum) & mask_s1
            best[improve] = curr_s1[improve]
            best_sum[improve] = sum_s1[improve]
            optimal_action_full[t][improve] = 2 * win_g + 1  # Gk-S1

        # welfare_mat[i, g, s] = welfare of group g under scenario "group s wins".
        N_grid = int(np.prod(shape_per_group))
        welfare_mat = np.zeros((N_grid, m, m), dtype=np.float64)
        for s in range(m):
            ws = welfare_scenarios[s]                     # shape_per_group + (m,)
            flat = ws.reshape(N_grid, m)
            for g in range(m):
                welfare_mat[:, g, s] = flat[:, g]

        if m == 3:
            try:
                p_mat = _solve_m3_via_subprocess(
                    welfare_mat, group_sizes, epsilon,
                    timeout_sec=int(os.environ.get("M3_SOLVER_TIMEOUT", "60")),
                )
            except M3SolverTimeout as _mt_exc:
                # Timeout: fall back to water-filling so the pipeline completes.
                warnings.warn(
                    f"m=3 analytical LP solver timed out ({_mt_exc}); "
                    f"falling back to water-filling for this grid "
                    f"({N_grid} rows)."
                )
                p_mat = np.zeros((N_grid, m), dtype=np.float64)
                for i in range(N_grid):
                    total_per_scenario = welfare_mat[i].sum(axis=0)
                    phi = _water_filling_per_capita(total_per_scenario, group_sizes, epsilon)
                    p_mat[i] = phi
        elif _HAS_LINPROG:
            p_mat = solve_optimization_mg_lp(welfare_mat, group_sizes, epsilon)
        else:
            p_mat = np.zeros((N_grid, m), dtype=np.float64)
            for i in range(N_grid):
                total_per_scenario = welfare_mat[i].sum(axis=0)
                phi = _water_filling_per_capita(total_per_scenario, group_sizes, epsilon)
                p_mat[i] = phi

        # Reshape p_mat back to per-axis grid + (m,)
        p_grid = p_mat.reshape(shape_per_group + (m,))
        optimal_p[t] = p_grid

        # SW[t] = sum_s p_s * welfare_scenarios[s]
        SW[t] = np.zeros_like(fv_idle)
        for s in range(m):
            SW[t] += p_grid[..., s][..., None] * welfare_scenarios[s]

    if not compute_payment:
        phi_star = np.zeros((S, T, m), dtype=np.float64)
        V_star = np.zeros((S, T), dtype=np.float64)
        for s_idx in range(S):
            s_flat = states[s_idx]
            idx = tuple(int(s_flat[2 * k + 1]) for k in range(m))
            for t in range(T):
                for g in range(m):
                    phi_star[s_idx, t, g] = optimal_p[t][idx + (g,)]
                V_star[s_idx, t] = float(SW[t][idx].sum())
        return _pack_result(states, phi_star, V_star, None, group_sizes,
                            num_states=2, T=T, delta=delta, epsilon=epsilon,
                            valuations=valuations, matrix0=matrix0, matrix1=matrix1)

    # No-i single-buyer VCG payments (baseline parity).

    # Compute pay_g_grid for each g_miss vectorized.

    def _compute_pay_g_vectorized(
        g_miss: int,
        optimal_action_full_local: np.ndarray,
    ) -> np.ndarray:
        sizes_no = _no_i_economy_indices(group_sizes, g_miss)
        M_idle_no, M_win_s0_no, M_win_s1_no = _build_no_i_transitions_K2(
            sizes_no, [matrix0[k] for k in range(m)],
            [matrix1[k] for k in range(m)])
        policy_no, W_no, _idle = _no_i_backward_induction_K2_mg(
            T, delta, valuations,
            M_idle_no, M_win_s0_no, M_win_s1_no, sizes_no)
        # Only cells where the full winner is g_miss.
        # For S1 winners shift the no-i axis down by one; S0 winners need no shift.
        pay_grid = np.zeros((T,) + shape_per_group, dtype=np.float64)
        for t in range(T):
            ac_full = optimal_action_full_local[t]
            act_g_full = ac_full // 2
            act_s_full = ac_full % 2
            mask_winner = (act_g_full == g_miss)
            if not mask_winner.any():
                continue
            mask_full_s1 = mask_winner & (act_s_full == 1)
            mask_full_s0 = mask_winner & (act_s_full == 0)

            shape_no = (sizes_no[0] + 1,) + tuple(sizes_no[k] + 1 for k in range(1, m)) if m == 1 else \
                       tuple(sizes_no[k] + 1 for k in range(m))

            def _shift_axis(arr, axis, shift):
                """Shift array along axis by shift positions, padding with fill."""
                fill = -1 if np.issubdtype(arr.dtype, np.integer) else np.nan
                if shift > 0:
                    pad_shape = list(arr.shape)
                    pad_shape[axis] = shift
                    pad = np.full(pad_shape, fill, dtype=arr.dtype)
                    return np.concatenate([pad, arr[..., :arr.shape[axis] - shift]], axis=axis)
                else:
                    return arr

            pol_shifted = _shift_axis(policy_no[t], g_miss, +1)  # boundary = -1
            # Pad pol_shifted to shape shape_per_group (with -1 at the
            # boundary that would be invalid).
            pad_shape = list(pol_shifted.shape)
            pad_shape[g_miss] = shape_per_group[g_miss]
            pol_padded = np.full(pad_shape, -1, dtype=policy_no.dtype)
            slicer = [slice(None)] * pol_shifted.ndim
            slicer[g_miss] = slice(0, pol_shifted.shape[g_miss])
            pol_padded[tuple(slicer)] = pol_shifted
            # For s1 winners: look up pol_padded (which shifted S1 axis by -1).
            ac_no_s1 = np.where(mask_full_s1, pol_padded, -1)
            # For s0 winners: no shift, just look up policy_no[t] (padded).
            pad_shape_no = list(policy_no[t].shape)
            pad_shape_no[g_miss] = shape_per_group[g_miss]
            pol_no_padded = np.full(pad_shape_no, -1, dtype=policy_no.dtype)
            slicer = [slice(None)] * policy_no[t].ndim
            slicer[g_miss] = slice(0, policy_no[t].shape[g_miss])
            pol_no_padded[tuple(slicer)] = policy_no[t]
            ac_no_s0 = np.where(mask_full_s0, pol_no_padded, -1)
            ac_no = np.where(mask_full_s1, ac_no_s1, ac_no_s0)

            # term1: v[ac_g_no][ac_s_no] when ac_no != -1
            term1 = np.zeros(shape_per_group, dtype=np.float64)
            nz = ac_no >= 0
            ag_no = ac_no // 2
            asg_no = ac_no % 2
            term1 = np.where(nz, valuations[ag_no, asg_no], 0.0)

            # term2: delta * (W_no[w*] - W_no[idle]) at idx_no.
            # W_no[t] has shape shape_no + (m,). Sum last dim to get total welfare.
            W_total = W_no[t].sum(axis=-1)  # shape shape_no
            # Build the shifted version of W_total (axis g_miss shifted by +1, pad with NaN).
            W_shifted = _shift_axis(W_total, g_miss, +1)
            W_shifted_padded = np.full(shape_per_group, np.nan, dtype=W_total.dtype)
            slicer = [slice(None)] * W_shifted.ndim
            slicer[g_miss] = slice(0, W_shifted.shape[g_miss])
            W_shifted_padded[tuple(slicer)] = W_shifted
            # No-shift version of W_total
            W_no_padded = np.full(shape_per_group, np.nan, dtype=W_total.dtype)
            W_no_padded_slicer = [slice(None)] * W_total.ndim
            W_no_padded_slicer[g_miss] = slice(0, W_total.shape[g_miss])
            W_no_padded[tuple(W_no_padded_slicer)] = W_total
            W_at_idx_no = np.where(mask_full_s1, W_shifted_padded, W_no_padded)

            # Total no-i welfare.
            fv_idle = _no_i_einsum_future(M_idle_no, W_no[t + 1], delta)  # shape shape_no + (m,)
            fv_idle_total = fv_idle.sum(axis=-1)  # shape shape_no
            # Pad idle total to full shape.
            fv_idle_padded = np.full(shape_per_group, np.nan, dtype=fv_idle_total.dtype)
            fv_idle_padded_slicer = [slice(None)] * fv_idle_total.ndim
            fv_idle_padded_slicer[g_miss] = slice(0, fv_idle_total.shape[g_miss])
            fv_idle_padded[tuple(fv_idle_padded_slicer)] = fv_idle_total
            fv_idle_at_idx_no = fv_idle_padded

            w_opt = np.where(mask_winner, W_at_idx_no, 0.0)
            w_act = np.where(mask_winner, fv_idle_at_idx_no, 0.0)
            term2 = delta * (w_opt - w_act)

            pay_grid[t] = np.where(mask_winner, term1 + term2, 0.0)
        return pay_grid

    pay_g_grid_per_g = np.zeros((m, T) + shape_per_group, dtype=np.float64)
    for g_miss in range(m):
        if group_sizes[g_miss] == 0:
            continue
        pay_g_grid_per_g[g_miss] = _compute_pay_g_vectorized(
            g_miss, optimal_action_full)

    phi_star = np.zeros((S, T, m), dtype=np.float64)
    V_star = np.zeros((S, T), dtype=np.float64)
    p_star = np.zeros((S, T, m), dtype=np.float64)

    for s_idx in range(S):
        s_flat = states[s_idx]
        idx = tuple(int(s_flat[2 * k + 1]) for k in range(m))
        for t in range(T):
            for g in range(m):
                phi_star[s_idx, t, g] = optimal_p[t][idx + (g,)]
            V_star[s_idx, t] = float(SW[t][idx + (0,)] + SW[t][idx + (1,)]) \
                if m == 2 else float(SW[t][idx].sum())
            # Expected no-i single-buyer VCG payment.
            # payment is already 0 when the winner is not g, so the
            # phi_g multiplier only affects the winner branch.
            for g in range(m):
                p_star[s_idx, t, g] = phi_star[s_idx, t, g] * float(pay_g_grid_per_g[g, t][idx])

    return _pack_result(states, phi_star, V_star, p_star, group_sizes,
                         num_states=2, T=T, delta=delta, epsilon=epsilon,
                         valuations=valuations, matrix0=matrix0, matrix1=matrix1,
                         optimal_action_full=optimal_action_full)



def _fair_pivot_bi_Kg_m2(
    group_sizes: List[int],
    num_states: int,
    T: int,
    delta: float,
    epsilon: float,
    valuations: np.ndarray,
    matrix0: Union[np.ndarray, List[np.ndarray]],
    matrix1: Union[np.ndarray, List[np.ndarray]],
    compute_payment: bool = True,
) -> Dict:
    """K in {3,4}, m=2: compressed compositional state DP."""
    assert num_states in (3, 4)
    assert len(group_sizes) == 2

    states = enumerate_market_states(group_sizes, num_states)
    S = states.shape[0]

    trans = precompute_general_transitions(group_sizes, matrix0, matrix1, num_states)
    trans0, trans1 = trans[0], trans[1]
    mapper0, mapper1 = trans0["mapper"], trans1["mapper"]
    n0 = mapper0.num_states_total
    n1 = mapper1.num_states_total

    V = np.zeros((T + 1, n0, n1), dtype=np.float64)
    V_g0 = np.zeros((T + 1, n0, n1), dtype=np.float64)
    V_g1 = np.zeros((T + 1, n0, n1), dtype=np.float64)
    p0_grid = np.zeros((T, n0, n1), dtype=np.float64)

    # Payment-only history grids are omitted for training-data generation.
    W0_g0_grid = np.zeros((T, n0, n1), dtype=np.float64) if compute_payment else None
    W0_g1_grid = np.zeros((T, n0, n1), dtype=np.float64) if compute_payment else None
    W1_g0_grid = np.zeros((T, n0, n1), dtype=np.float64) if compute_payment else None
    W1_g1_grid = np.zeros((T, n0, n1), dtype=np.float64) if compute_payment else None

    v0 = valuations[0]
    v1 = valuations[1]

    for t in range(T - 1, -1, -1):
        V_next = V[t + 1]
        V_next_g0 = V_g0[t + 1]
        V_next_g1 = V_g1[t + 1]

        E_idle = delta * (trans0["idle"] @ V_next @ trans1["idle"].T)
        E_idle_g0 = delta * (trans0["idle"] @ V_next_g0 @ trans1["idle"].T)
        E_idle_g1 = delta * (trans0["idle"] @ V_next_g1 @ trans1["idle"].T)

        W0 = E_idle.copy()
        W0_g0 = E_idle_g0.copy()
        W0_g1 = E_idle_g1.copy()
        for s in range(num_states):
            T_win = trans0["win"][s]
            if T_win is None:
                continue
            E_future = delta * (T_win @ V_next @ trans1["idle"].T)
            E_val = v0[s] + E_future
            E_future_g0 = delta * (T_win @ V_next_g0 @ trans1["idle"].T)
            E_val_g0 = v0[s] + E_future_g0
            E_future_g1 = delta * (T_win @ V_next_g1 @ trans1["idle"].T)
            E_val_g1 = E_future_g1

            valid_rows = (T_win.sum(axis=1) > 0.5)
            valid_mask = valid_rows[:, None] & np.ones((1, n1), dtype=bool)
            update_mask = valid_mask & (E_val > W0)
            W0[update_mask] = E_val[update_mask]
            W0_g0[update_mask] = E_val_g0[update_mask]
            W0_g1[update_mask] = E_val_g1[update_mask]

        W1 = E_idle.copy()
        W1_g0 = E_idle_g0.copy()
        W1_g1 = E_idle_g1.copy()
        for s in range(num_states):
            T_win = trans1["win"][s]
            if T_win is None:
                continue
            E_future = delta * (trans0["idle"] @ V_next @ T_win.T)
            E_val = v1[s] + E_future
            E_future_g0 = delta * (trans0["idle"] @ V_next_g0 @ T_win.T)
            E_val_g0 = E_future_g0
            E_future_g1 = delta * (trans0["idle"] @ V_next_g1 @ T_win.T)
            E_val_g1 = v1[s] + E_future_g1

            valid_cols = (T_win.sum(axis=1) > 0.5)
            valid_mask = np.ones((n0, 1), dtype=bool) & valid_cols[None, :]
            update_mask = valid_mask & (E_val > W1)
            W1[update_mask] = E_val[update_mask]
            W1_g0[update_mask] = E_val_g0[update_mask]
            W1_g1[update_mask] = E_val_g1[update_mask]

        # Solve optimal p0
        w0_flat = W0.flatten()
        w1_flat = W1.flatten()
        w0_g0_flat = W0_g0.flatten()
        w0_g1_flat = W0_g1.flatten()
        w1_g0_flat = W1_g0.flatten()
        w1_g1_flat = W1_g1.flatten()

        p0_flat = solve_analytical_p0_vectorized(
            w0_g0_flat, w1_g0_flat, w0_g1_flat, w1_g1_flat,
            tuple(group_sizes), epsilon)
        val_flat = p0_flat * w0_flat + (1 - p0_flat) * w1_flat

        p0_grid[t] = p0_flat.reshape(n0, n1)
        V[t] = val_flat.reshape(n0, n1)
        V_g0[t] = p0_grid[t] * W0_g0 + (1 - p0_grid[t]) * W1_g0
        V_g1[t] = p0_grid[t] * W0_g1 + (1 - p0_grid[t]) * W1_g1

        if compute_payment:
            # Save per-group welfare only when exact VCG labels are requested.
            W0_g0_grid[t] = W0_g0
            W0_g1_grid[t] = W0_g1
            W1_g0_grid[t] = W1_g0
            W1_g1_grid[t] = W1_g1

    # Map compressed-state (idx0, idx1) results back to expanded (S, m*K) rows
    phi_star = np.zeros((S, T, 2), dtype=np.float64)
    V_star = np.zeros((S, T), dtype=np.float64)
    p_star = np.zeros((S, T, 2), dtype=np.float64) if compute_payment else None

    # Build lookup: expanded row -> compressed (idx0, idx1)
    combo0_to_idx = {c: i for i, c in enumerate(mapper0.states)}
    combo1_to_idx = {c: i for i, c in enumerate(mapper1.states)}

    for s_idx in range(S):
        row = states[s_idx]
        # expanded row layout: [g0_s0, g0_s1, ..., g0_s_{K-1},
        #                       g1_s0, g1_s1, ..., g1_s_{K-1}]
        combo0 = tuple(int(row[s]) for s in range(num_states))
        combo1 = tuple(int(row[num_states + s]) for s in range(num_states))
        idx0 = combo0_to_idx.get(combo0)
        idx1 = combo1_to_idx.get(combo1)
        if idx0 is None or idx1 is None:
            continue
        for t in range(T):
            p0 = float(p0_grid[t, idx0, idx1])
            phi_star[s_idx, t, 0] = p0
            phi_star[s_idx, t, 1] = 1.0 - p0
            V_star[s_idx, t] = float(V[t, idx0, idx1])
            if compute_payment:
                # NOTE: this path uses group-level VCG as a placeholder;
                # it is not the no-i single-buyer VCG used for K=2.
                pay_g0 = float(W1_g1_grid[t, idx0, idx1] - W1_g0_grid[t, idx0, idx1])
                pay_g1 = float(W0_g0_grid[t, idx0, idx1] - W0_g1_grid[t, idx0, idx1])
                p_star[s_idx, t, 0] = p0 * pay_g0
                p_star[s_idx, t, 1] = (1.0 - p0) * pay_g1

    return _pack_result(states, phi_star, V_star, p_star, group_sizes,
                         num_states=num_states, T=T, delta=delta, epsilon=epsilon,
                         valuations=valuations, matrix0=matrix0, matrix1=matrix1)



def _fair_pivot_bi_Kg_mg(
    group_sizes: List[int],
    num_states: int,
    T: int,
    delta: float,
    epsilon: float,
    valuations: np.ndarray,
    matrix0: Union[np.ndarray, List[np.ndarray]],
    matrix1: Union[np.ndarray, List[np.ndarray]],
    compute_payment: bool = True,
) -> Dict:
    """K>=3, m>=3. Combines Path B grid LP with compressed K state."""
    raise NotImplementedError(
        "K>2 and m>2 simultaneously is not in the 9-experiment scope; "
        "extend Path D by combining _TransitionMatrixBuilder with m-group LP.")



def num_states_check(matrix: np.ndarray, expected: int) -> bool:
    return matrix.shape == (expected, expected)


def _axis_mask(shape_per_group: Tuple[int, ...], axis: int,
                op: str, threshold: int) -> np.ndarray:
    """
    Boolean mask of shape `shape_per_group` that is True where the
    index along `axis` satisfies `arange(shape[axis]) op threshold`.
    """
    out = np.zeros(shape_per_group, dtype=bool)
    idx = np.arange(shape_per_group[axis])
    if op == "<":
        mask_1d = idx < threshold
    elif op == ">":
        mask_1d = idx > threshold
    elif op == "<=":
        mask_1d = idx <= threshold
    elif op == ">=":
        mask_1d = idx >= threshold
    else:
        raise ValueError(f"unknown op: {op}")
    # Broadcast along axis
    slicer = [slice(None)] * len(shape_per_group)
    slicer[axis] = mask_1d
    out[tuple(slicer)] = True
    return out


def _water_filling_per_capita(total_welfare: np.ndarray,
                               group_sizes: List[int],
                               epsilon: float) -> np.ndarray:
    """Fallback for K=2, m>=3 when linprog is not available."""
    return solve_eq1_one(
        sw=total_welfare, sw_remaining=np.zeros_like(total_welfare),
        group_sizes=group_sizes, epsilon=epsilon)


def _pack_result(states, phi_star, V_star, p_star, group_sizes,
                  num_states, T, delta, epsilon,
                  valuations, matrix0, matrix1,
                  optimal_action_full=None) -> Dict:
    # Normalize to lists for downstream consumption
    if isinstance(matrix0, np.ndarray):
        matrix0_list = [matrix0] * len(group_sizes)
    else:
        matrix0_list = list(matrix0)
    if isinstance(matrix1, np.ndarray):
        matrix1_list = [matrix1] * len(group_sizes)
    else:
        matrix1_list = list(matrix1)
    result = {
        "states":       states,
        "phi":          phi_star,
        "V":            V_star,
        "group_sizes":  np.asarray(group_sizes, dtype=np.int64),
        "num_states":   num_states,
        "T":            T,
        "delta":        delta,
        "epsilon":      epsilon,
        "valuations":   valuations,
        "matrix0":      matrix0_list,
        "matrix1":      matrix1_list,
    }
    if p_star is not None:
        result["payment"] = p_star
    if optimal_action_full is not None:
        result["optimal_action_full"] = optimal_action_full
    return result
