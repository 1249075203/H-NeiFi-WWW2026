from network import TrajEncoder
from torch.optim import Adam
from arguments import Args
import numpy as np
import torch
from env import *
from reward_function import episode_policy_num
from algorithm import policy_gradient
import copy
import random
import matplotlib.pyplot as plt

args = Args()
agent_num = args.agent_num
buffer_size = args.buffer_size
Gamma = args.gamma

plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 15})

# test_seed = 1
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# set_seed()

def init_hidden(num_layers, batch, h_dim,device):
    c = torch.zeros(num_layers, batch, h_dim).to(device)
    h = torch.zeros(num_layers, batch, h_dim).to(device)
    return (h, c)


def explore_action(op, epsiode):
    if type(op) != list:
        op = [op]
    for i in range(len(op)):
        op[i] = op[i] + np.random.uniform(-0.1, 0.1)
    return op


def visit_agent_2(agent_all, episode):
    fig, ax = plt.subplots()
    x = [i for i in range(len(agent_all))]
    plt.xticks(range(len(x)), x)
    for i in range(agent_num):
        y = []
        for j in range(len(agent_all)):
            y.append(agent_all[j][i])
        ax.plot(x, y, marker='d')
    plt.savefig(f'test/fig_agent_{episode}.png')
    plt.clf()
    plt.cla()
    plt.close("all")


def visit_agent(agent_all, agent_all_exp, episode, agent_all_noRL):
    print(agent_all_noRL)
    while (len(agent_all) <= args.max_step):
        agent_all.append(agent_all[-1])
        agent_all_exp.append(agent_all_exp[-1])

    while (len(agent_all_noRL) <= args.max_step):
        agent_all_noRL.append(agent_all_noRL[-1])

    # while (len(IWPI_opinion) <= args.max_step):
    #     IWPI_opinion.append(IWPI_opinion[-1])
    fig, ax = plt.subplots()
    x = [i for i in range(len(agent_all))]
    # plt.xticks(range(len(x)), x)

    for i in range(agent_num):
        y = []
        for j in range(len(agent_all)):
            y.append(agent_all[j][i])
        ax.plot(x, y, marker='>', color='y',label='Normal people with ACP')


    for i in range(agent_num):
        if agent_all_noRL != []:
        # for i in range(2):
            y = []
            for j in range(len(agent_all_noRL)):
                y.append(agent_all_noRL[j][i])
            ax.plot(x, y, color='g',label='Normal people without ACP')

    for i in range(len(agent_all_exp[0])):
        y = []
        for j in range(len(agent_all_exp)):
            y.append(agent_all_exp[j][i])
        ax.plot(x, y, marker='o', ls='--',label='Experts')
    plt.xlabel("k")
    plt.ylabel("x")
    plt.axis([0, len(x), 0, args.opinions_end])
    plt.grid(True)
    plt.xticks(np.arange(0, len(x) + 1, step=5))
    plt.yticks(np.arange(0, args.opinions_end + 1, step=2))

    plt.savefig(f'result/fig_agent_{episode}.jpg', dpi=600, bbox_inches='tight', format='jpg')
    plt.clf()
    plt.cla()
    plt.close("all")

def get_nei_mar2(aop):
    a_ma = []
    for i in range(len(aop)):
        temp = []
        for j in range(len(aop)):
            if abs(aop[i] - aop[j]) <= 0.01:
                temp.append(1)
            else:
                temp.append(0)
        a_ma.append(temp)
    return a_ma


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
def get_nei_mar_noself(aop):
    a_ma = []
    for i in range(len(aop)):
        temp = []
        for j in range(len(aop)):
            if abs(aop[i] - aop[j]) <= 0.01:
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

def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x / np.sum(e_x, axis=0)

def plot_loss(loss):
    x = [i for i in range(len(loss))]
    y = loss
    plt.plot(x,y)
    plt.savefig(f'Loss/loss.jpg', dpi=600, bbox_inches='tight', format='jpg')

# opinion communication without ACP
def no_RL_main(agent_opinions, agent_confident, agent_exp, agent_exp_confident, episode):
    agent_all = []
    agent_all_exp = []


    group_all = []

    agent_leader = [0] * len(agent_opinions)

    for i in range(len(agent_exp)):
        sub_group = []
        sub_group.append([i])
        sub_group.append(i)
        group_all.append(sub_group)



    group_num = 0
    for num_agent in range(len(agent_opinions)):
        min_exp = 0
        min_distance = args.opinions_end
        for current_exp_idx in range(len(agent_exp)):
            current_distance = abs(agent_opinions[num_agent] - agent_exp[current_exp_idx])
            if current_distance <= min_distance:
                min_distance = current_distance
                min_exp = current_exp_idx
        group_all[min_exp].append(num_agent)


    for i in range(len(group_all)):
        for agent_idx in group_all[i][2:]:
            agent_leader[agent_idx] = i

    for _ in range(args.max_step):


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

        agent_opinions_next = copy.deepcopy(agent_opinions)

        for i in range(args.agent_num):
            agent_i_confident = agent_confident[i]
            agent_i_exp = group_all[agent_leader[i]][0]
            agent_exp_opinion = 0
            agent_nei_num = 0
            agent_nei_opinion = 0

            i_agent_nei_info, i_agent_nei_info_index = get_neigbor_info(group_all[agent_leader[i]][2:],
                                                                        agent_opinions, i)
            for j in i_agent_nei_info:
                agent_nei_opinion += j
                agent_nei_num += 1

            for j in agent_i_exp:
                agent_exp_opinion += agent_exp[j]
            agent_exp_opinion = float(agent_exp_opinion / len(agent_i_exp))


            if agent_nei_num != 0:
                agent_nei_opinion_all = args.weight_normal * (
                            agent_nei_opinion / agent_nei_num) + args.weight_exp * agent_exp_opinion
                agent_opinions_next[i] = agent_i_confident * agent_opinions[i] + (
                            1 - agent_i_confident) * agent_nei_opinion_all

            else:
                agent_opinions_next[i] = agent_opinions[i]

        agent_exp_next = copy.deepcopy(agent_exp)
        for i in range(len(agent_exp)):
            exp_nei_opinions = 0
            exp_nei_num = 0
            good_nei = []



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


        agent_all.append(agent_opinions)
        agent_all_exp.append(agent_exp)

        agent_opinions = agent_opinions_next
        agent_opinions[-1] = 3.5
        agent_exp = agent_exp_next

        if is_done(agent_opinions):
            break

    agent_all.append(agent_opinions)
    agent_all_exp.append(agent_exp)
    temp_cu_m = get_nei_mar_noself(agent_opinions)
    temp_cu_m = np.matrix(temp_cu_m)
    # cu_num = calc_cluster_num(temp_cu_m)
    # print('CC:', cu_num)

    return agent_all

# H-NeiFi main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    encoder = TrajEncoder(args, args.input_dim).to(device)
    optimizer = Adam(encoder.parameters(), lr=args.learning_rate)
    test_seed = 1
    std_all = []  # CD
    episode_reward = []

    # start training
    buffer_total = []

    LOSS_ALL = []
    for episode in range(args.training_episodes):
        print(episode)
        ALL_time = []
        if episode >= 150:
            set_seed(test_seed)
            test_seed = test_seed+1

        # Reverse action selection probability
        if episode <= 10:
            noisy = 0.3
        elif episode <= 50:
            noisy = 0.2
        elif episode <= 100:
            noisy = 0.1
        else:
            noisy = 0

        agent_all = []
        agent_all_exp = []

        # Initialize non-experts
        agent_opinions = np.linspace(args.opinions_begin, args.opinions_end, args.agent_num,
                                     dtype=np.float32).tolist()

        # Malicious agents
        # agent_opinions[-1] = 3.5

        # Uneven distributed
        # agent_opinions = []
        # for i in range(args.agent_num):
        #     opinion_i = random.uniform(args.opinions_begin, args.opinions_end)
        #     agent_opinions.append(opinion_i)
        # agent_opinions.sort()

        # agent_opinions = []
        # for i in range(10):
        #     opinion_i = random.uniform(0, 1)
        #     agent_opinions.append(opinion_i)
        # for i in range(30):
        #     opinion_i = random.uniform(2, 4)
        #     agent_opinions.append(opinion_i)
        # agent_opinions.sort()

        # stubbornnes of non-experts
        agent_confident = [0] * args.agent_num
        for i in range(args.agent_num):
            agent_confident[i] = random.uniform(0.4, 0.6)

        # Initialize experts
        agent_exp = args.agent_exp

        # stubbornnes of experts
        agent_exp_confident = [0] * len(agent_exp)
        for i in range(len(agent_exp_confident)):
            agent_exp_confident[i] = random.uniform(0.8, 0.9)

        agent_all_noRL = no_RL_main(agent_opinions, agent_confident, agent_exp, agent_exp_confident, episode)


        #subgrouops
        group_all = []
        agent_leader = [0] * len(agent_opinions)

        for i in range(len(agent_exp)):
            sub_group = []
            sub_group.append([i])
            sub_group.append(i)
            group_all.append(sub_group)

        # Initialize subgroups
        for num_agent in range(len(agent_opinions)):
            min_exp = 0
            min_distance = args.opinions_end
            for current_exp_idx in range(len(agent_exp)):
                current_distance = abs(agent_opinions[num_agent] - agent_exp[current_exp_idx])
                if current_distance <= min_distance:
                    min_distance = current_distance
                    min_exp = current_exp_idx
            group_all[min_exp].append(num_agent)


        for i in range(len(group_all)):
            for agent_idx in group_all[i][2:]:
                agent_leader[agent_idx] = i

        state_all = []
        buffer = []

        for _ in range(agent_num):
            temp = []
            buffer.append(temp)
            state_all.append(init_hidden(2 * encoder.num_layers, 1, encoder.h_dim,device))

        special_agent = []

        for _ in range(args.max_step):
            print("step:", _)

            LK_LSTM = []
            D_T = []  #

            reward_g3_flag = 0  # Not enabled, ignored

            # local goal
            sub_opinions = []
            for sub_group in group_all:
                sb_exp = sub_group[0]
                sb_exp_opinions = 0
                for sb_exp_idx in sb_exp:
                    sb_exp_opinions += agent_exp[sb_exp_idx]
                sub_opinions.append(float(sb_exp_opinions / len(sb_exp)))

            # Subgroup merging
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

            starter.record()
            # ACP
            for i in range(args.agent_num):
                D_i = []
                i_agent_nei_info, i_agent_nei_info_index = get_neigbor_info(group_all[agent_leader[i]][2:],
                                                                            agent_opinions, i)


                if i_agent_nei_info == []:
                    temp_LK = [0 for i in range(len(agent_opinions))]
                    LK_LSTM.append(temp_LK)
                    special_agent.append(i)
                    D_T.append(D_i)
                    continue


                shuffle_nei = list(zip(i_agent_nei_info, i_agent_nei_info_index))
                random.shuffle(shuffle_nei)
                i_agent_nei_info[:], i_agent_nei_info_index[:] = zip(*shuffle_nei)


                agent_i_target = sub_opinions[agent_leader[i]]

                agent_i_position = agent_opinions[i] - agent_i_target
                if -0.1 <= agent_i_position <= 0.1:
                    agent_i_position = 0.1

                input_i_all = []

                # S_i(k)
                for agent_input_i in range(len(i_agent_nei_info)):
                    temp = []
                    temp.append([abs(i_agent_nei_info[agent_input_i] - agent_i_target) / abs(agent_i_position)])
                    temp.append([abs(i_agent_nei_info[agent_input_i] - Args.target_opinions) / abs(agent_opinions[i] - Args.target_opinions)])

                    temp.append([abs(agent_i_position)])
                    temp.append([abs(agent_opinions[i] - Args.target_opinions)])


                    input_i_all.append(temp)


                D_i.append(get_aik(i_agent_nei_info, i_agent_nei_info_index))
                D_i.append(i_agent_nei_info)
                input_tensor = torch.from_numpy(np.array(input_i_all, dtype=np.float32)).reshape((-1, len(input_i_all), Args.input_dim)).to(device)
                output, new_state = encoder(input_tensor, state_all[i], episode, False)

                output_list = output.detach().cpu().numpy().squeeze().tolist()
                output_temp = output.detach().cpu().numpy().squeeze().tolist()
                if type(output_temp) != list:
                    output_temp = [output_temp]

                if type(output_list) != list:
                    output_list = [output_list]

                avg_output = np.mean(output_temp)
                temp_LK = [0 for i in range(len(agent_opinions))]

                # Reverse Action Selection
                for nei in range(len(i_agent_nei_info_index)):
                    random_flag = random.uniform(0, 1)
                    if random_flag >= noisy:
                        current_action = output_temp[nei]
                        if current_action >= avg_output:
                            temp_LK[i_agent_nei_info_index[nei]] = 1
                            output_list[nei] = 1
                        else:
                            temp_LK[i_agent_nei_info_index[nei]] = 0
                            output_list[nei] = 0
                    if random_flag < noisy:
                        current_action = output_temp[nei]
                        if current_action >= avg_output:
                            temp_LK[i_agent_nei_info_index[nei]] = 0
                            output_list[nei] = 0
                        else:
                            temp_LK[i_agent_nei_info_index[nei]] = 1
                            output_list[nei] = 1
                output_list = torch.tensor(output_list).float().to(device)
                D_i.append([output, output_list])
                state_all[i] = new_state

                D_i.append(new_state)
                D_T.append(D_i)

                LK_LSTM.append(temp_LK)

            #UPDATE OPINION
            agent_opinions_next = copy.deepcopy(agent_opinions)
            for i in range(args.agent_num):
                agent_i_confident = agent_confident[i]
                agent_i_exp = group_all[agent_leader[i]][0]
                agent_exp_opinion = 0
                agent_nei_num = 0
                agent_nei_opinion = 0
                for j in range(args.agent_num):
                    if LK_LSTM[i][j] == 1:
                        agent_nei_opinion += agent_opinions[j]
                        agent_nei_num += 1

                for j in agent_i_exp:
                    agent_exp_opinion += agent_exp[j]
                agent_exp_opinion = float(agent_exp_opinion / len(agent_i_exp))



                if agent_nei_num != 0:
                    agent_nei_opinion_all = args.weight_normal * (agent_nei_opinion / agent_nei_num) + args.weight_exp * agent_exp_opinion
                    agent_opinions_next[i] = agent_i_confident * agent_opinions[i] + (1 - agent_i_confident) * agent_nei_opinion_all
                else:
                    agent_opinions_next[i] = agent_opinions[i]
            ender.record()
            torch.cuda.synchronize()
            gpu_time = starter.elapsed_time(ender) / 1000
            ALL_time.append(gpu_time)
            # print(f"Total GPU time: {gpu_time:.2f} seconds")

            # PCP
            agent_exp_next = copy.deepcopy(agent_exp)
            for i in range(len(agent_exp)):
                exp_nei_opinions = 0
                exp_nei_num = 0
                good_nei = []

                current_exp_position = agent_exp[i] - args.target_opinions

                for j in range(len(agent_exp)):
                    if i == j:
                        continue
                    if abs((agent_exp[i] + agent_exp[j])/2 - args.target_opinions) <= abs(current_exp_position):
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

            for i in range(agent_num):
                if i in special_agent:
                    continue

                reward = episode_policy_num(i, agent_opinions, agent_opinions_next, sub_opinions[agent_leader[i]],
                                            reward_g3_flag)
                episode_reward.append(reward)
                D_T[i].append(reward)
                buffer[i].append(D_T[i])

            agent_all.append(agent_opinions)
            agent_all_exp.append(agent_exp)

            if is_done(agent_opinions):
                print("OURS_CS:", _)
                miss_value = cal_miss(agent_opinions, Args.target_opinions)
                print("OURTS_SCD:", miss_value)
                break

            if _ == args.max_step-1 :
                print("OURS_CS:", _)
                miss_value = cal_miss(agent_opinions, Args.target_opinions)
                print("OURTS_SCD:", miss_value)

            agent_opinions = agent_opinions_next
            agent_opinions[-1] = 3.5

            agent_exp = agent_exp_next


        gpu_time = sum(ALL_time)/len(ALL_time)
        print(f"Total GPU time: {gpu_time:.4f} seconds")

        buffer_new = []
        for i in range(agent_num):
            if i in special_agent:
                continue
            else:
                buffer_new.append(buffer[i])

        agent_all.append(agent_opinions)
        agent_all_exp.append(agent_exp)
        for i in buffer_new:
            if len(buffer_total) >= args.buffer_size:
                del buffer_total[0]
            buffer_total.append(i)

        # TRAIN MARL
        if len(buffer_total) <= args.batch_size:
            loss = policy_gradient(buffer_total, encoder, optimizer, len(buffer_total) - 1,device)
        else:
            loss = policy_gradient(buffer_total, encoder, optimizer, args.batch_size,device)
        print(loss)
        loss = loss.detach().cpu().numpy()
        LOSS_ALL.append(loss)
        temp_cu_m = get_nei_mar_noself(agent_opinions)
        temp_cu_m = np.matrix(temp_cu_m)
        cu_num = calc_cluster_num(temp_cu_m)
        print('CC:', cu_num)
        print('=======================================')
        visit_agent(agent_all, agent_all_exp, episode, agent_all_noRL)


if __name__ == '__main__':
    main()
