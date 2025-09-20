# -*- coding: utf-8 -*-

import numpy as np                  # 数组
import matplotlib.pyplot as plt     # 数据可视化的库
import random
import time
from arguments import Args
import copy
args = Args()
plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 15})

def cal_miss(agent_opinions,target):
    miss = 0
    for i in agent_opinions:
        miss += abs(i - target)
    miss = miss / len(agent_opinions)
    return miss
def calc_cluster_num(A):
    node_lst = list(range(len(A)))
    edge_lst = []

    for i in node_lst:
        for j in node_lst:  # 结尾点
            # if i != j and j in agent_objs[i].workfor:
            if i != j and A[j, i] == 1:
                edge_lst.append([i, j])

    # 初始化前导节点 pre 数据结构
    pre = [i for i in range(len(node_lst))]
    # 遍历边集合，合并
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
        r = pre[r]  # 找到前导节点
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
            if i == j:
                temp.append(0)
            else:
                if abs(aop[i]-aop[j]) <= 0.01:
                    temp.append(1)
                else:
                    temp.append(0)
        a_ma.append(temp)
    return a_ma
def compare_exp(exp_a, exp_b):
    if len(exp_a) == len(exp_b):
        diff = list(set(exp_a).difference(set(exp_b)))
        if diff == []:
            return True
        else:
            return False
    else:
        return False

def is_stable(agent_value, th=1e-2):
    C = np.zeros(shape=(args.agent_num, args.agent_num), dtype=np.float32)
    for i in range(args.agent_num):  # 找每个i的邻居，并更新每个i
        for j in range(args.agent_num):
            if i != j:
                distance = abs(agent_value[i] - agent_value[j])
                if distance < r_c and distance>th:  # 在差值r_c以内的为邻居
                    return False
    return True
# num = 40        # 智能体个数
# interval = 4       # 智能体的间隔
# exp_opinion = interval/2
# exp_values = [exp_opinion for i in range(36)]
r_c = args.r_c
m = 1       # 共同邻居选几个
beta = 0    # 远邻居可调参数

for episode in range(50):
    print(episode)
    agent_t = np.zeros(shape=(args.agent_num,), dtype=np.int)
    agent_value = np.linspace(args.opinions_begin, args.opinions_end, args.agent_num, dtype=np.float32)
    # agent_value = np.random.random_sample(num) * interval
    # plt.scatter(agent_t, agent_value)

    agent_confident = [0] * args.agent_num
    for i in range(args.agent_num):
        agent_confident[i] = random.uniform(0.4, 0.6)

    agent_exp = args.agent_exp
    agent_exp_confident = [0] * len(agent_exp)
    for i in range(len(agent_exp_confident)):
        agent_exp_confident[i] = random.uniform(0.8, 0.9)
    total_values_exp = [agent_exp]
    # weight_exp = 0
    # weight_nei = 1


    total_values = [agent_value]

    # 更新
    test_times = 0
    FLOPs = 0
    start_time = time.time()

    group_all = []
    agent_leader = [0] * len(agent_value)

    for i in range(len(agent_exp)):
        sub_group = []
        sub_group.append([i])
        sub_group.append(i)
        group_all.append(sub_group)


    for num_agent in range(len(agent_value)):
        min_exp = 0
        min_distance = args.opinions_end
        for current_exp_idx in range(len(agent_exp)):
            current_distance = abs(agent_value[num_agent] - agent_exp[current_exp_idx])
            if current_distance <= min_distance:
                min_distance = current_distance
                min_exp = current_exp_idx
        group_all[min_exp].append(num_agent)


    for i in range(len(group_all)):
        for agent_idx in group_all[i][2:]:
            agent_leader[agent_idx] = i



    while True: # [1, test_times]

        test_times += 1

        next_agent_value = np.zeros(shape=(args.agent_num,), dtype=np.float32)
        avg_agent_value = np.mean(agent_value)

        sub_opinions = []
        for sub_group in group_all:
            sb_exp = sub_group[0]
            sb_exp_opinions = 0
            for sb_exp_idx in sb_exp:
                sb_exp_opinions += agent_exp[sb_exp_idx]
            sub_opinions.append(float(sb_exp_opinions / len(sb_exp)))


        for group_i in range(len(sub_opinions) - 1):
            for group_j in range(group_i + 1, len(sub_opinions)):
                if abs(sub_opinions[group_i] - sub_opinions[group_j]) <= args.sg_c:

                    group_i_exp = group_all[group_i][0]
                    group_j_exp = group_all[group_j][0]
                    compare_flag = compare_exp(group_i_exp, group_j_exp)
                    if compare_flag == False:

                        new_group_exp = group_i_exp + group_j_exp
                        new_group_nei = group_all[group_i][2:] + group_all[group_j][2:]
                        new_group = [new_group_exp, group_all[group_i][1]]

                        for new_nei in new_group_nei:
                            new_group.append(new_nei)

                        for i in range(len(group_all)):
                            if group_all[i][1] == group_all[group_i][1] or group_all[i][1] == group_all[group_j][1]:
                                group_all[i] = new_group


        sub_opinions = []
        for sub_group in group_all:
            sb_exp = sub_group[0]
            sb_exp_opinions = 0
            for sb_exp_idx in sb_exp:
                sb_exp_opinions += agent_exp[sb_exp_idx]
            sub_opinions.append(float(sb_exp_opinions / len(sb_exp)))


        N_is = []
        for i in range(args.agent_num):
            N_i = []
            for j in range(args.agent_num):
                distance = abs(agent_value[i] - agent_value[j])
                if distance < r_c:
                    N_i.append(j)
            N_is.append(set(N_i))


        selected_N_is = []
        for i in range(args.agent_num):
            selected_N_i = list(N_is[i].copy())
            other_N_i = set(range(args.agent_num)) - N_is[i]
            FLOPs += 1
            p_ij = []

            # print(len(selected_N_i))
            # print(len(other_N_i))
            if len(selected_N_i) != args.agent_num:
                fenmu = 0
                for j in other_N_i:
                    fenmu += np.power(np.e, beta * len(N_is[j] & N_is[i]))
                    FLOPs += 2
                for j in other_N_i:
                    fenzi = np.power(np.e, beta * len(N_is[j] & N_is[i]))
                    p_ij.append(fenzi/fenmu)
                    FLOPs += 3

                if m == 1:
                    # print(p_ij)
                    # if len(p_ij) != 0:
                    idx = np.argwhere(p_ij == np.amax(p_ij)).squeeze(1)
                    selected_idx = idx[np.random.choice(len(idx), 1)]   # 均匀随机选1个
                    FLOPs += 2
                    selected_N_i.append(selected_idx)
                    selected_N_is.append(selected_N_i)
                    # else:
                    #     selected_N_is.append(selected_N_i)

            else:
                if m == 1:
                    selected_N_is.append(selected_N_i)


        for i in range(args.agent_num):
            neighbour_num = 0
            neighbour_value = 0
            agent_i_exp = group_all[agent_leader[i]][0]
            agent_exp_opinion = 0

            for j in selected_N_is[i]:
                # print('j',j)
                neighbour_value += agent_value[j]
                FLOPs += 1

            for j in agent_i_exp:
                agent_exp_opinion += agent_exp[j]
            agent_exp_opinion = float(agent_exp_opinion / len(agent_i_exp))



            nei_opinion = args.weight_normal * (neighbour_value/len(selected_N_is[i])) + args.weight_exp * agent_exp_opinion
            next_agent_value[i] = agent_value[i] * agent_confident[i] + (1-agent_confident[i]) * nei_opinion

            FLOPs += 1


        # 将临时变量赋值给原始的数组
        agent_value = next_agent_value
        total_values.append(agent_value)

        agent_exp_next = copy.deepcopy(agent_exp)
        for i in range(len(agent_exp)):
            exp_nei_opinions = 0
            exp_nei_num = 0
            good_nei = []
            bad_nei = []

            current_exp_position = agent_exp[i] - args.target_opinions

            for j in range(len(agent_exp)):
                if i == j:
                    continue
                if abs((agent_exp[i] + agent_exp[j]) / 2 - args.target_opinions) <= abs(current_exp_position):
                    good_nei.append([abs(agent_exp[j] - agent_exp[i]), j])

            good_nei = sorted(good_nei, key=(lambda x: x[0]), reverse=False)

            if good_nei == []:
                agent_exp_next[i] = agent_exp[i]
            else:
                for nei in good_nei:
                    exp_nei_opinions += agent_exp[nei[1]]
                    exp_nei_num += 1
                    temp_next = agent_exp_confident[i] * agent_exp[i] + (1 - agent_exp_confident[i]) * (
                            exp_nei_opinions / exp_nei_num)
                    if abs(temp_next - agent_exp[i]) >= 0.5:
                        break
                agent_exp_next[i] = agent_exp_confident[i] * agent_exp[i] + (1 - agent_exp_confident[i]) * (
                        exp_nei_opinions / exp_nei_num)
        agent_exp = agent_exp_next
        total_values_exp.append(agent_exp_next)

        if is_stable(agent_value):
            temp_ma = get_nei_mar2(agent_value)
            temp_ma = np.matrix(temp_ma)
            # print(temp_ma)
            cu_num = calc_cluster_num(temp_ma)
            print('CC:', cu_num)
            miss_values = cal_miss(agent_value, Args.target_opinions)
            print("SCD:", abs(miss_values))
            print("CS:", test_times)
            break

        if test_times == 35:
            # print('time:', test_times)
            # temp_ma = get_nei_mar2(agent_value)
            # temp_ma = np.matrix(temp_ma)
            # # print(temp_ma)
            # cu_num = calc_cluster_num(temp_ma)
            # print('蔟:', cu_num)
            # miss_values = cal_miss(agent_value, Args.target_opinions)
            # print("距离目标的误差:", abs(miss_values))
            # print("步数", test_times)
            break

    end_time = time.time()
    # print("spending {} s time".format(end_time-start_time))


    fig, ax = plt.subplots()
    # print("test_times:", test_times)
    # print("total_values:", len(total_values))
    # test_times = test_times + test_times//5 + 1     # 补充1/5的时间
    test_times = 36
    plt.axis([0, test_times, 0, args.opinions_end])
    plt.xlabel("k")
    plt.ylabel("x")
    plt.grid(True)
    plt.xticks(np.arange(0, test_times, step=5))
    plt.yticks(np.arange(0, args.opinions_end + 1, step=2))

    for _ in range(test_times-len(total_values)):
        total_values.append(
            total_values[-1]
        )
    for _ in range(test_times-len(total_values_exp)):
        total_values_exp.append(
            total_values_exp[-1]
        )

    total_values = np.array(total_values)
    total_values_exp = np.array(total_values_exp)

    # print("test_times:", test_times)
    # print("total_values:", len(total_values))
    for i in range(args.agent_num):
        ax.plot(list(range(test_times)), total_values[:, i], marker='>', color='y')
    for i in range(len(agent_exp)):
        ax.plot(list(range(test_times)), total_values_exp[:, i] , marker='o', ls='--', label='Experts')
    plt.savefig(f'result_CNR/{args.agent_num}-{args.opinions_end}-{episode}.jpg', dpi=600, bbox_inches='tight', format='jpg')
    # episode = 1
    # plt.savefig(f'C:\\Users\\blacksheep\\Desktop\\P\\CNR/eps/fig_agent_{episode}.eps',dpi=600,bbox_inches='tight',format='eps')
    # plt.savefig(f'C:\\Users\\blacksheep\\Desktop\\P\\CNR/pdf/fig_agent_{episode}.pdf', dpi=600,bbox_inches='tight',format='pdf')
    # plt.savefig(f'C:\\Users\\blacksheep\\Desktop\\P\\CNR/svg/fig_agent_{episode}.svg', dpi=600,bbox_inches='tight',format='svg')
    # plt.savefig(f'C:\\Users\\blacksheep\\Desktop\\P\\CNR/jpg/fig_agent_{episode}.jpg', dpi=600, bbox_inches='tight', format='jpg')
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close("all")
    print("====================================================")