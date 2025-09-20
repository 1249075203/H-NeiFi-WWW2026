import torch
from torch import nn


def make_mlp(dim_list, bias_list, act_list, drop_list):
    num_layers = len(dim_list) - 1
    layers = []
    for i in range(num_layers):
        dim_in, dim_out, bias, activation, drop_prob = dim_list[i], dim_list[i + 1], bias_list[i], act_list[i], \
        drop_list[i]
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        if drop_prob > 0:
            layers.append(nn.Dropout(p=drop_prob))
    return nn.Sequential(*layers)


class TrajEncoder(nn.Module):
    '''
    Encode past trajectory using a Multi-Layer Perceptron (MLP)
    while keeping the same input and output interface as the original LSTM-based encoder.
    '''

    def __init__(self, args, input_dim=1):
        super(TrajEncoder, self).__init__()
        self.input_dim = input_dim
        self.h_dim = args.h_dim
        self.emb_dim = args.emb_dim
        self.begin_rate = 0.2
        self.num_layers = 1


        self.pos_emb = make_mlp(
            dim_list=[self.input_dim, self.emb_dim],
            bias_list=[True],
            act_list=['relu'],
            drop_list=[0]
        )


        mlp_hidden_dims = [self.emb_dim, 64, 128, 64]
        self.encoder = make_mlp(
            dim_list=mlp_hidden_dims + [self.h_dim],
            bias_list=[True] * len(mlp_hidden_dims) + [True],
            act_list=['relu'] * len(mlp_hidden_dims) + ['none'],
            drop_list=[0] * (len(mlp_hidden_dims) + 1)
        )


        self.last_fc = nn.Linear(self.h_dim, 1)
        self.ac = nn.Softmax(dim=-1)
    def forward(self, in_traj_ego, last_state, episode, random_last_fc=False):
        batch_size = in_traj_ego.size(0)
        seq_len = in_traj_ego.size(1)

        in_traj_embedding = self.pos_emb(in_traj_ego.view(-1, self.input_dim))

        output = self.encoder(in_traj_embedding)
        if random_last_fc:
            ran_para = 0
            recover_w = self.last_fc.weight.data
            self.last_fc.weight.data = self.last_fc.weight.data + ran_para * self.begin_rate

        output = self.last_fc(output).view(batch_size, seq_len, -1).squeeze(-1)
        output = self.ac(output)

        if random_last_fc:
            self.last_fc.weight.data = recover_w
        return output, (torch.zeros_like(last_state[0]), torch.zeros_like(last_state[1]))
