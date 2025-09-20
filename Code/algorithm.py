import torch
import random
from arguments import Args
import numpy as np
import copy
from torch.distributions.categorical import Categorical
args =Args()
# batch_size = args.batch_size
gamma = args.gamma

#policy gradient
def policy_gradient(D, model, optimizer,batch_size,device):
    random_chos_D = random.sample(range(1,len(D)),batch_size)
    obs = []        #sik
    acts = []       #aik
    rewards = []    #rik
    loss_all = torch.tensor([1e-6], dtype=torch.float32,device=device)
    for i in random_chos_D:
        temp_obs_0 = []
        temp_obs = []
        temp_acts = []
        temp_rewards = []
        for j in D[i]:
            temp_obs_0.append(j[0])
            temp_obs.append(j[1])
            temp_acts.append(j[2])
            temp_rewards.append(j[4])
        obs.append(temp_obs)
        acts.append(temp_acts)
        rewards.append(temp_rewards)

    for i in range(len(obs)):
        loss_all = torch.tensor([1e-6], dtype=torch.float32,device=device)
        discounted_ep_r = np.zeros_like(rewards[i])
        running_add = 0
        for t in reversed(range(0, len(rewards[i]))):
            running_add = running_add * gamma + rewards[i][t]
            discounted_ep_r[t] = running_add
        discounted_ep_r = discounted_ep_r - np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)
        discounted_ep_rs_norm = discounted_ep_r
        reward_tensor = torch.FloatTensor(discounted_ep_rs_norm).to(device)

        for j in range(len(obs[i])):
            output = acts[i][j]

            output_prob = output[0]
            output_action = output[1]    

            action_prob = output_prob * output_action + (1 - output_prob) * (1 - output_action)
            log_prob = torch.log(action_prob + 1e-8)

            # log_prob = torch.log(output[0])
            # log_prob = log_prob * output[1]
            selected_log_probs = reward_tensor[j] * log_prob
            loss = -selected_log_probs.mean()
            if torch.isnan(loss):
                print('here')
                print(output)
                print(loss)
            loss.requires_grad_(True)
            loss = loss
            loss_all += loss
    optimizer.zero_grad()
    loss_all /= len(obs[i])
    loss_all.backward(retain_graph=True)
    optimizer.step()

    return loss_all