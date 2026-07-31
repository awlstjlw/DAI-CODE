import numpy as np


def solve_analytical_p0(
    welfare_g0_tuple: tuple,
    welfare_g1_tuple: tuple,
    welfare_g2_tuple: tuple,
    group_sizes: tuple,
    epsilon: float
) -> tuple:
    """Analytical p0 solver (external API, original logic preserved)."""
    def extract_vals(w_tuple):
        return [np.array([x], dtype=np.float64) for x in w_tuple]

    w_g0_vecs = extract_vals(welfare_g0_tuple)
    w_g1_vecs = extract_vals(welfare_g1_tuple)
    w_g2_vecs = extract_vals(welfare_g2_tuple)

    p_matrix = solve_optimization_3g_vectorized(
        w_g0_vecs, w_g1_vecs, w_g2_vecs,
        group_sizes, epsilon
    )
    return tuple(p_matrix[0])


def _solve_single_lp(c, V_norm, epsilon):
    """Pure-Python 3-variable LP solver via 2-D vertex enumeration."""
    c = np.asarray(c, dtype=np.float64)
    V_norm = np.asarray(V_norm, dtype=np.float64)

    # Eliminate p2 = 1 - p0 - p1; constant c2 does not affect argmax.
    obj_a = c[0] - c[2]
    obj_b = c[1] - c[2]

    # 2-D constraints in (p0, p1): a*p0 + b*p1 <= rhs.
    constraints = [
        (-1.0, 0.0, 0.0),   # p0 >= 0
        (0.0, -1.0, 0.0),   # p1 >= 0
        (1.0, 1.0, 1.0),    # p2 >= 0  => p0 + p1 <= 1
        (1.0, 0.0, 1.0),    # p0 <= 1
        (0.0, 1.0, 1.0),    # p1 <= 1
        (-1.0, -1.0, 0.0),  # p2 <= 1
    ]

    # Fairness constraints: |V_norm_i @ p - V_norm_j @ p| <= epsilon.
    # Substitute p2 = 1 - p0 - p1, then express both inequalities in (p0, p1).
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        a = (V_norm[i, 0] - V_norm[i, 2]) - (V_norm[j, 0] - V_norm[j, 2])
        b = (V_norm[i, 1] - V_norm[i, 2]) - (V_norm[j, 1] - V_norm[j, 2])
        const = V_norm[i, 2] - V_norm[j, 2]
        constraints.append((a, b, epsilon - const))
        constraints.append((-a, -b, epsilon + const))

    constraints = np.array(constraints, dtype=np.float64)
    n = len(constraints)

    best_val = -np.inf
    best_p = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float64)
    feasible_found = False

    for i in range(n):
        for j in range(i + 1, n):
            a1, b1, c1 = constraints[i]
            a2, b2, c2 = constraints[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-12:
                continue
            p0 = (c1 * b2 - c2 * b1) / det
            p1 = (a1 * c2 - a2 * c1) / det
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

    if not feasible_found:
        return best_p

    best_p = np.clip(best_p, 0.0, 1.0)
    s = np.sum(best_p)
    if s > 0:
        best_p = best_p / s
    else:
        best_p = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float64)
    return best_p


def solve_optimization_3g_vectorized(
    w_g0_scenarios: list,
    w_g1_scenarios: list,
    w_g2_scenarios: list,
    group_sizes: tuple,
    epsilon: float
) -> np.ndarray:
    """Vectorized 3-group optimizer using pure-Python vertex enumeration."""
    N0, N1, N2 = group_sizes
    N0 = np.float64(max(N0, 1e-9))
    N1 = np.float64(max(N1, 1e-9))
    N2 = np.float64(max(N2, 1e-9))

    def safe_cast(arr):
        arr = np.array(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    w_g0_scenarios = [safe_cast(w) for w in w_g0_scenarios]
    w_g1_scenarios = [safe_cast(w) for w in w_g1_scenarios]
    w_g2_scenarios = [safe_cast(w) for w in w_g2_scenarios]

    batch_size = w_g0_scenarios[0].shape[0]
    results = np.zeros((batch_size, 3), dtype=np.float64)

    c0 = -(w_g0_scenarios[0] + w_g1_scenarios[0] + w_g2_scenarios[0])
    c1 = -(w_g0_scenarios[1] + w_g1_scenarios[1] + w_g2_scenarios[1])
    c2 = -(w_g0_scenarios[2] + w_g1_scenarios[2] + w_g2_scenarios[2])

    for i in range(batch_size):
        try:
            c = np.array([c0[i, 0], c1[i, 0], c2[i, 0]], dtype=np.float64)

            V = np.zeros((3, 3), dtype=np.float64)
            V[0, :] = [w_g0_scenarios[0][i, 0], w_g0_scenarios[1][i, 0], w_g0_scenarios[2][i, 0]]
            V[1, :] = [w_g1_scenarios[0][i, 0], w_g1_scenarios[1][i, 0], w_g1_scenarios[2][i, 0]]
            V[2, :] = [w_g2_scenarios[0][i, 0], w_g2_scenarios[1][i, 0], w_g2_scenarios[2][i, 0]]

            V_norm = np.zeros_like(V, dtype=np.float64)
            V_norm[0] = V[0] / N0
            V_norm[1] = V[1] / N1
            V_norm[2] = V[2] / N2

            p_vals = _solve_single_lp(c, V_norm, float(epsilon))
            results[i] = p_vals
        except Exception as e:
            print(f"[WARNING] batch {i} solver failed: {str(e)[:100]}")
            results[i] = [1.0/3.0, 1.0/3.0, 1.0/3.0]

    results = np.clip(results, 0.0, 1.0)
    results = results / np.sum(results, axis=1, keepdims=True)
    return results
