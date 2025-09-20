import torch
from torch import nn

def make_mlp(dim_list, bias_list, act_list, drop_list):
    num_layers = len(dim_list) - 1
    layers = []
    for i in range(num_layers):
        # layer info
        dim_in, dim_out, bias, activation, drop_prob = dim_list[i], dim_list[i+1], bias_list[i], act_list[i], drop_list[i]
        # linear layer
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))
        # add activation
        if (activation == 'relu'):
            layers.append(nn.ReLU())
        elif (activation == 'sigmoid'):
            layers.append(nn.Sigmoid())
        elif (activation == 'tanh'):
            layers.append(nn.Tanh())
        # dropout layer
        if (drop_prob > 0):
            layers.append(nn.Dropout(p=drop_prob))
    return nn.Sequential(*layers)

def init_hidden(num_layers, batch, h_dim):
    c = torch.zeros(num_layers, batch, h_dim)
    h = torch.zeros(num_layers, batch, h_dim)
    return (h, c)

class TrajEncoder(nn.Module):
    '''
    Encode past trajectory using a unidirectional LSTM
    '''
    def __init__(self, args, input_dim=1):
        super(TrajEncoder, self).__init__()
        self.input_dim = input_dim
        self.h_dim = args.h_dim
        self.emb_dim = args.emb_dim
        self.begin_rate = 0.2
        self.num_layers = 1
        # 使用 make_mlp 创建位置嵌入层
        self.pos_emb = make_mlp(
            dim_list=[self.input_dim, self.emb_dim],
            bias_list=[True],
            act_list=['relu'],
            drop_list=[0]
        )


        self.encoder = nn.LSTM(self.emb_dim, self.h_dim, self.num_layers, bidirectional=False, batch_first=True, dropout=0)
        self.last_fc = nn.Linear(self.h_dim, 1)
        self.ac = nn.Softmax(dim=-1)
    def forward(self, in_traj_ego,last_state,episode,random_last_fc=False):
        if random_last_fc:
            # ran_para = nn.Parameter(torch.randn([1, 72]))
            ran_para = 0
            recover_w = self.last_fc.weight.data
            self.last_fc.weight.data = self.last_fc.weight.data + ran_para * self.begin_rate
            neighbor_number = in_traj_ego.size(1)
            in_traj_embedding = self.pos_emb(in_traj_ego.view(1, neighbor_number, self.input_dim))
            in_traj_embedding = in_traj_embedding.view(1, neighbor_number, self.emb_dim)
            # encoding by an LSTM
            output, new_state = self.encoder(in_traj_embedding, last_state)
            output = self.last_fc(output).squeeze()
            output = self.ac(output)
            self.last_fc.weight.data = recover_w
            return output, new_state


        else:
            neighbor_number = in_traj_ego.size(1)
            in_traj_embedding = self.pos_emb(in_traj_ego.view(1, neighbor_number, self.input_dim))
            in_traj_embedding = in_traj_embedding.view(1, neighbor_number, self.emb_dim)
            # encoding by an LSTM
            output, new_state = self.encoder(in_traj_embedding, last_state)
            output = self.last_fc(output).squeeze()
            output = self.ac(output)
            return output, new_state
