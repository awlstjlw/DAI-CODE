import numpy as np

def solve_analytical_p0(
    welfare_g0_tuple: tuple,
    welfare_g1_tuple: tuple,
    group_sizes: tuple,
    epsilon: float
) -> float:
    g0_val_if_g0_wins, g0_val_if_g1_wins = welfare_g0_tuple
    g1_val_if_g0_wins, g1_val_if_g1_wins = welfare_g1_tuple
    
    res = solve_analytical_p0_vectorized(
        np.array([g0_val_if_g0_wins]), 
        np.array([g0_val_if_g1_wins]),
        np.array([g1_val_if_g0_wins]), 
        np.array([g1_val_if_g1_wins]),
        group_sizes, epsilon
    )
    return float(res[0])

def solve_analytical_p0_vectorized(
    w_g0_if_g0_wins: np.ndarray,
    w_g0_if_g1_wins: np.ndarray,
    w_g1_if_g0_wins: np.ndarray,
    w_g1_if_g1_wins: np.ndarray,
    group_sizes: tuple,
    epsilon: float
) -> np.ndarray:
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