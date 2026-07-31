import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config_m3 import AuctionConfigM3


@dataclass
class Group:
    id: int
    matrix0: np.ndarray
    matrix1: np.ndarray
    valuations: np.ndarray
    init_prob: np.ndarray
    agents_idx: List[int] = field(default_factory=list)

    @property
    def states_size(self):
        return len(self.valuations)


@dataclass
class Agent:
    id: int
    label: int
    state_probs: np.ndarray
    cur_state: int


@dataclass
class OtherData:
    agents_size: int
    group_sizes: List[int] = field(default_factory=list)
    calc_group_sizes: List[int] = field(default_factory=list)
    auc_group_sizes: List[int] = field(default_factory=list)
    agents_calc: List[Agent] = field(default_factory=list)
    agents_auc: List[Agent] = field(default_factory=list)


def generate_initial_probability() -> np.ndarray:
    return np.array([0.5, 0.5])


def generate_data_m3(
    config: Optional[AuctionConfigM3] = None,
    epsilon=0,
    seed_of_value=0,
    seed_of_matrix=0,
    seed_of_grouping=0,
    specified_group_sizes: Optional[List[int]] = None,
):
    """Generate three-group auction data exactly as in experiment 3."""
    if config is None:
        config = AuctionConfigM3()

    config.reset_data_seed(seed_of_value, seed_of_matrix, seed_of_grouping)
    config.reset_fairness_epsilon(epsilon)

    group_sizes = list(specified_group_sizes or [20, 20, 20])
    if len(group_sizes) != 3:
        raise ValueError(f"m=3 generator requires exactly three group sizes: {group_sizes}")

    config.NUM_AGENTS = sum(group_sizes)
    config.CALC_GROUP_SIZES = group_sizes.copy()

    valuations = config.generate_random_value(config.SEED_OF_VALUE)
    all_matrices = config.generate_random_matrix(6, config.SEED_OF_MATRIX)
    groups = [
        Group(
            id=i,
            matrix0=all_matrices[i * 2],
            matrix1=all_matrices[i * 2 + 1],
            valuations=valuations[i],
            init_prob=generate_initial_probability(),
        )
        for i in range(3)
    ]

    labels = np.concatenate(
        [np.full(count, group_id, dtype=np.int64)
         for group_id, count in enumerate(group_sizes)]
    )
    rng = np.random.default_rng(seed_of_grouping)
    rng.shuffle(labels)

    agents = []
    for agent_id, group_id_raw in enumerate(labels):
        group_id = int(group_id_raw)
        agent = Agent(
            id=agent_id,
            label=group_id,
            state_probs=groups[group_id].init_prob.copy(),
            cur_state=0,
        )
        agents.append(agent)
        groups[group_id].agents_idx.append(agent_id)

    actual_sizes = [len(group.agents_idx) for group in groups]
    other_data = OtherData(
        agents_size=len(agents),
        group_sizes=actual_sizes,
        calc_group_sizes=[0, 0, 0],
        auc_group_sizes=[0, 0, 0],
    )

    for group_id, group_size in enumerate(actual_sizes):
        for index in range(group_size):
            original_agent_id = groups[group_id].agents_idx[index]
            agent = agents[original_agent_id]
            other_data.agents_calc.append(copy.deepcopy(agent))
            other_data.calc_group_sizes[agent.label] += 1
            other_data.agents_auc.append(copy.deepcopy(agent))
            other_data.auc_group_sizes[agent.label] += 1

    return groups, agents, other_data, config
