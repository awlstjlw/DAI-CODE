import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from config import AuctionConfig
import copy

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

def generate_data(config = None, epsilon=0, seed_of_value=0, seed_of_matrix=0, seed_of_grouping = 0, specified_group_sizes: Optional[List[int]] = None) -> tuple[List[Group], List[Agent], OtherData, AuctionConfig]:
    if config is None:
        config = AuctionConfig()
        
    config.reset_data_seed(seed_of_value, seed_of_matrix, seed_of_grouping)
    config.reset_fairness_epsilon(epsilon)
        
    num_agents = config.NUM_AGENTS
    
    groups = []
    tmp_valuations = config.generate_random_value(config.SEED_OF_VALUE)
    
    all_mat = config.generate_random_matrix(4, config.SEED_OF_MATRIX)
    for i in range(2):
        groups.append(Group(
            id=i,
            matrix0=all_mat[i * 2],
            matrix1=all_mat[i * 2 + 1],
            valuations=tmp_valuations[i],
            init_prob=generate_initial_probability()
        ))
    
    agents = []
    
    rng = np.random.default_rng(seed_of_grouping) 
    
    if specified_group_sizes is not None:
        total_specified = sum(specified_group_sizes)
        if total_specified != num_agents:
            print(f"Warning: The sum of specified group sizes ({total_specified}) is inconsistent with config.NUM_AGENTS ({num_agents}).")
            print(f"Automatically adjusted total number of agents to {total_specified}.")
            num_agents = total_specified
            config.NUM_AGENTS = num_agents
            
        labels = []
        for g_id, count in enumerate(specified_group_sizes):
            labels.extend([g_id] * count)
        labels = np.array(labels)
        
    else:
        labels = [0, 1] * (num_agents // 2)
        remainder = rng.choice(2, size=num_agents % 2, replace=False)
        labels = np.concatenate([labels, remainder])
    
    rng.shuffle(labels)
    labels = labels.tolist()
    
    for i in range(num_agents):
        group_id = labels[i]
        agent = Agent(
            id=i,
            label=group_id,
            state_probs=groups[group_id].init_prob.copy(),
            cur_state=0
        )
        agents.append(agent)
        groups[group_id].agents_idx.append(i)
    
    other_data = OtherData(
        agents_size=len(agents),
        group_sizes=[len(groups[0].agents_idx), len(groups[1].agents_idx)],
        calc_group_sizes=[0, 0],
        auc_group_sizes=[0, 0],
        agents_calc=[],
        agents_auc=[]
    )
    
    for idx, group_size_each in enumerate(other_data.group_sizes):
        for i in range(group_size_each):
            original_agent_id = groups[idx].agents_idx[i]
            agent = agents[original_agent_id]
            
            other_data.agents_calc.append(copy.deepcopy(agent))
            other_data.calc_group_sizes[agent.label] += 1
            
            other_data.agents_auc.append(copy.deepcopy(agent))
            other_data.auc_group_sizes[agent.label] += 1
    
    save_calc_group_sizes(other_data.calc_group_sizes)
    
    return groups, agents, other_data, config

def save_calc_group_sizes(calc_group_sizes: List[int]):
    """Save only the calculation group sizes to file"""
    with open("calc_group_sizes.txt", 'w', encoding='utf-8') as f:
        f.write(f"{calc_group_sizes[0]},{calc_group_sizes[1]}")