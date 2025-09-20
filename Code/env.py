from arguments import Args
import numpy as np
import copy
import math
import random


arg = Args()
agent_num = arg.agent_num

def get_neigbor_info(agent_group_nei,agent_opinions,i):

    current_agent = agent_opinions[i]
    nei_info = []
    nei_info_index = []
    for j in range(len(agent_opinions)):
        if i==j:
            continue
        if abs(agent_opinions[j] - current_agent) <= Args.r_c:
            nei_info.append(agent_opinions[j])
            nei_info_index.append(j)

    return nei_info, nei_info_index

def get_aik(neibor_value, index):
    aik = [0] * agent_num
    for i in range(len(neibor_value)):
        aik[index[i]] = neibor_value[i]
    return aik

def get_nei_info_LSTM(output, i_nei_info_index):
    i_nei_LSTM_index = []
    for i in range(len(output)):
        if output[i] >= 0.5:
            i_nei_LSTM_index.append(i_nei_info_index[i])
    return i_nei_LSTM_index


def get_LK_i_2(i_nei_index,i):
    Ni_num = 0
    if i not in i_nei_index:
        Ni_num = 1
    Ni_num += len(i_nei_index)
    LK_i  = [0] * agent_num
    for nei in i_nei_index:
        LK_i[nei] = 1/Ni_num
    LK_i[i] = 1/Ni_num
    return LK_i

def get_LK_i(i_nei_index):
    Ni_num = len(i_nei_index)
    LK_i = [0] * arg.agent_num
    for nei in i_nei_index:
        LK_i[nei] = 1/Ni_num
    return LK_i

def is_done(agent_value, th=1e-2):
    for i in range(Args.agent_num):
        for j in range(Args.agent_num):
            if i != j:
                distance = abs(agent_value[i] - agent_value[j])
                if distance < 1 and distance>th:
                    return False
    return True

def calc_cluster_num(A):
    node_lst = list(range(len(A)))
    edge_lst = []

    for i in node_lst:
        for j in node_lst:
            if i != j and A[j, i] == 1:
                edge_lst.append([i, j])


    pre = [i for i in range(len(node_lst))]

    for e_i in range(len(edge_lst)):
        edge = edge_lst[e_i]
        join(
            node_lst.index(edge[0]),
            node_lst.index(edge[1]),
            pre
        )
    groups = []
    for n_i in range(len(node_lst)):
        groups.append(find(n_i, pre))
    groups = np.array(groups)

    cluster_num = np.unique(groups).shape[0]
    return cluster_num
def find(x, pre):
    r = x
    while pre[r] != r:
        r = pre[r]
    i = x
    while i != r:
        j = pre[i]
        pre[i] = r
        i = j
    return r
def join(x, y, pre):
    a = find(x, pre)
    b = find(y, pre)
    if a != b:
        pre[a] = b
def get_nei_mar2(aop):
    a_ma = []
    for i in range(len(aop)):
        temp = []
        for j in range(len(aop)):
            if abs(aop[i]-aop[j]) <= 1:
                temp.append(1)
            else:
                temp.append(0)
        a_ma.append(temp)
    return a_ma


def make_new_connet(LK):
    new_LK = copy.deepcopy(LK)
    for i in range(len(LK[0])):
        avg_action = 0
        nei_num = 0
        for j in range(len(LK[0])):
            if LK[i][j] != 0:
                avg_action += LK[i][j]
                nei_num += 1
        avg_action = avg_action/nei_num

        for j in range(len(LK[0])):
            if LK[i][j] >= avg_action:
                new_LK[i][j] = 1
            else:
                new_LK[i][j] = 0

    return new_LK

def get_nei_RL(output,i_agent_nei_info, i_agent_nei_info_index,current_opinion,agent_i_target):
    temp_nei = []
    for i in range(len(i_agent_nei_info_index)):
        nei_position = i_agent_nei_info[i] - agent_i_target
        temp_nei.append([nei_position,i_agent_nei_info_index[i]])

    temp_nei = sorted(temp_nei,key=(lambda x:x[0]),reverse=False )

    down_nei = temp_nei [:len(i_agent_nei_info_index)]
    up_nei = temp_nei[len(i_agent_nei_info_index):]

    output_max = max(output)
    output_choice = output.index(output_max)
    if output_choice == 0:
        final_nei = down_nei
    else:
        final_nei = up_nei


    temp_LK = [0] * agent_num
    for i in final_nei:
        temp_LK[i[1]] = 1

    return temp_LK

def is_get(agent_opinions,target):
    miss_value = cal_miss(agent_opinions,target)
    if miss_value <= 0.1:
        return True
    else:
        return False

def cal_miss(agent_opinions,target):
    miss = 0
    for i in agent_opinions:
        miss += abs(i - target)
    miss = miss / len(agent_opinions)
    return miss