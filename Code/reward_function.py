import copy
import numpy as np
from arguments import Args

agent_num = Args.agent_num


def episode_policy_num(i,agent_opinions,agent_opinions_next,agent_target,reward_g3_flag):
    r_i_k = 0
    r_i_k_all = 0

    current_opinions = agent_opinions[i]
    next_opinions = agent_opinions_next[i]

    # local reward
    if abs(current_opinions - agent_target) >= 0.25:
        if abs(next_opinions - agent_target) <= abs(current_opinions - agent_target):
            r_i_k = abs(abs(next_opinions - agent_target) - abs(current_opinions - agent_target)) / abs(
                current_opinions - agent_target)

        if abs(next_opinions - agent_target) > abs(current_opinions - agent_target):
            r_i_k = -1 * abs(abs(next_opinions - agent_target) - abs(current_opinions - agent_target)) / abs(
             current_opinions - agent_target)
    if abs(current_opinions - agent_target) < 0.25:
        r_i_k = (0.25 - abs(next_opinions - agent_target)) / 0.25


    # global reward
    if current_opinions - Args.target_opinions >= 0.5:
        if abs(next_opinions - Args.target_opinions) <= abs(current_opinions - Args.target_opinions):
            r_i_k_all = abs(abs(next_opinions - Args.target_opinions) - abs(current_opinions - Args.target_opinions)) / abs(
                current_opinions - Args.target_opinions)

        if abs(next_opinions - Args.target_opinions) > abs(current_opinions -Args.target_opinions):
            r_i_k_all = -1 * abs(abs(next_opinions - Args.target_opinions) - abs(current_opinions - Args.target_opinions)) / abs(
                current_opinions - Args.target_opinions)

    if abs(current_opinions - Args.target_opinions) < 0.5:
        r_i_k_all = (0.5 - abs(next_opinions - Args.target_opinions)) / 0.5

    return 1 * r_i_k + 1 * r_i_k_all