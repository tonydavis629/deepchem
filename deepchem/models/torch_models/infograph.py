import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import math

try:
    from torch_geometric.nn import NNConv, Set2Set
except:
    pass
from deepchem.models.torch_models.torch_model import TorchModel

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(Encoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(11, 128), ReLU(), Linear(128, dim * dim)) # 11 edge features
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        # self.lin1 = torch.nn.Linear(2 * dim, dim)
        # self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.node_features))
        h = out.unsqueeze(0)

        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_features))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            # print(out.shape) : [num_node x dim]
            feat_map.append(out)

        out = self.set2set(out, data.graph_index)
        return out, feat_map[-1]
    
    
class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    
    
class InfoGraph(torch.nn.Module):
    def __init__(self, num_features, dim, use_unsup_loss=False, separate_encoder=False):
        super(InfoGraph, self).__init__()

        self.embedding_dim = dim
        self.separate_encoder = separate_encoder

        self.local = True
        # self.prior = False

        self.encoder = Encoder(num_features, dim)
        if separate_encoder:
            self.unsup_encoder = Encoder(num_features, dim)
            self.ff1 = FF(2*dim, dim)
            self.ff2 = FF(2*dim, dim)

        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc2 = torch.nn.Linear(dim, 1)

        if use_unsup_loss:
            self.local_d = FF(dim, dim)
            self.global_d = FF(2*dim, dim)
        
        self.init_emb()

    def init_emb(self):
      initrange = -1.5 / self.embedding_dim
      for m in self.modules():
          if isinstance(m, nn.Linear):
              torch.nn.init.xavier_uniform_(m.weight.data)
              if m.bias is not None:
                  m.bias.data.fill_(0.0)


    def forward(self, data):
        out, M = self.encoder(data)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        pred = out.view(-1)
        return pred

    def unsup_loss(self, data):
        if self.separate_encoder:
            y, M = self.unsup_encoder(data)
        else:
            y, M = self.encoder(data)
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        measure = 'JSD'
        if self.local:
            loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.graph_index, measure)
        return loss


    def unsup_sup_loss(self, data):
        y, M =   self.encoder(data)
        y_, M_ = self.unsup_encoder(data)

        g_enc = self.ff1(y)
        g_enc1 = self.ff2(y_)

        measure = 'JSD'
        loss = global_global_loss_(g_enc, g_enc1, data.edge_index, data.graph_index, measure)

        return loss

class InfographModel(TorchModel):
    pass

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs))#.cuda()
    neg_mask = torch.ones((num_nodes, num_graphs))#.cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
    E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
    E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

    return E_neg - E_pos

def global_global_loss_(g_enc, g_enc1, edge_index, batch, measure):
    '''
    Args:
        g: Global features
        g1: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]

    pos_mask = torch.eye(num_graphs)#.cuda()
    neg_mask = 1 - pos_mask

    res = torch.mm(g_enc, g_enc1.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
    E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
    E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

    return E_neg - E_pos

def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq

def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y