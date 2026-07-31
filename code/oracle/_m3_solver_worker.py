"""Standalone subprocess worker for the m=3 analytical LP solver."""
from __future__ import annotations
import sys, os, pickle

# Path setup when called from inside code/oracle/
_here = os.path.dirname(os.path.abspath(__file__))
_code_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _code_root)
sys.path.insert(0, os.path.dirname(_code_root))

import numpy as np
from code.oracle.fair_pivot_bi import solve_optimization_m3_lp


def main():
    data = pickle.load(sys.stdin.buffer)
    welfare = np.asarray(data['welfare_scenarios'], dtype=np.float64)
    group_sizes = data['group_sizes']
    epsilon = float(data['epsilon'])
    p = solve_optimization_m3_lp(welfare, group_sizes, epsilon)
    pickle.dump({'p': p.astype(np.float64)}, sys.stdout.buffer)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
