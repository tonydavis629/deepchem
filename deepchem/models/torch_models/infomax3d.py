import torch
from torch import nn
from deepchem.feat.molecule_featurizers.conformer_featurizer import full_atom_feature_dims, full_bond_feature_dims
from deepchem.models.torch_models.layers import MultilayerPerceptron
from math import sqrt
import dgl
import torch.nn.functional as F
import dgl.function as fn
from typing import Dict, List, Union, Callable
import numpy as np
from functools import partial


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for i, dim in enumerate(full_atom_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for i, embedder in enumerate(self.atom_embedding_list):
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                x_embedding += self.atom_embedding_list[i](x[:, i].long() + 1)
            else:
                x_embedding += self.atom_embedding_list[i](x[:, i].long())

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for i, dim in enumerate(full_bond_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if self.padding:
                bond_embedding += self.bond_embedding_list[i](
                    edge_attr[:, i].long() + 1)
            else:
                bond_embedding += self.bond_embedding_list[i](
                    edge_attr[:, i].long())

        return bond_embedding


# networks

class Net3D(nn.Module):

    def __init__(self,
                #  node_dim,
                #  edge_dim,
                 hidden_dim,
                 target_dim,
                 readout_aggregators: List[str],
                 batch_norm=False,
                 node_wise_output_layers=2,
                 readout_batchnorm=True,
                 batch_norm_momentum=0.1,
                 reduce_func='sum',
                 dropout=0.0,
                 propagation_depth: int = 4,
                 readout_layers: int = 2,
                 readout_hidden_dim=None,
                 fourier_encodings=0,
                 activation: str = 'SiLU',
                 update_net_layers=2,
                 message_net_layers=2,
                 use_node_features=False,
                 **kwargs):
        super(Net3D, self).__init__()
        self.fourier_encodings = fourier_encodings
        edge_in_dim = 3 if fourier_encodings == 0 else 2 * fourier_encodings + 1  # originally 1 XXX

        self.edge_input = nn.Sequential(
            MultilayerPerceptron(d_input=edge_in_dim,
                                 d_output=hidden_dim,
                                 d_hidden=(hidden_dim,),
                                 batch_norm=True,
                                 batch_norm_momentum=batch_norm_momentum),
            torch.nn.SiLU())

        self.use_node_features = use_node_features
        if self.use_node_features:
            self.atom_encoder = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
            nn.init.normal_(self.node_embedding)

        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(
                Net3DLayer(edge_dim=hidden_dim,
                           hidden_dim=hidden_dim,
                           batch_norm=batch_norm,
                           batch_norm_momentum=batch_norm_momentum,
                           dropout=dropout,
                           mid_activation=activation,
                           reduce_func=reduce_func,
                           message_net_layers=message_net_layers,
                           update_net_layers=update_net_layers))

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MultilayerPerceptron(
                d_input=hidden_dim,
                d_output=hidden_dim,
                d_hidden=(hidden_dim,),
                batch_norm=True,
                batch_norm_momentum=batch_norm_momentum)

        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators

        self.output = MultilayerPerceptron(
            d_input=hidden_dim * len(self.readout_aggregators),
            d_output=target_dim,
            d_hidden=(readout_hidden_dim,) *
            (readout_layers -
             1),  # -1 because the input layer is not considered a hidden layer
            batch_norm=readout_batchnorm,
            batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLGraph):
        if self.use_node_features:
            graph.ndata['feat'] = self.atom_encoder(graph.ndata['feat'])
        else:
            graph.ndata['feat'] = self.node_embedding[None, :].expand(
                graph.number_of_nodes(), -1)

        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(
                graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        if self.node_wise_output_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [
            dgl.readout_nodes(graph, 'feat', op=aggr)
            for aggr in self.readout_aggregators
        ]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_edge_func(self, edges):
        return {
            'd': F.silu(self.edge_input(edges.data['edge_attr']))
        }  # 'originally 'd'


class Net3DLayer(nn.Module):

    def __init__(self, edge_dim, reduce_func, hidden_dim, batch_norm,
                 batch_norm_momentum, dropout, mid_activation,
                 message_net_layers, update_net_layers):
        super(Net3DLayer, self).__init__()

        self.message_network = nn.Sequential(
            MultilayerPerceptron(d_input=hidden_dim * 2 + edge_dim,
                                 d_output=hidden_dim,
                                 d_hidden=(hidden_dim,) *
                                 (message_net_layers - 1),
                                 batch_norm=batch_norm,
                                 batch_norm_momentum=batch_norm_momentum,
                                 dropout=dropout), torch.nn.SiLU())
        if reduce_func == 'sum':
            self.reduce_func = fn.sum
        elif reduce_func == 'mean':
            self.reduce_func = fn.mean
        else:
            raise ValueError('reduce function not supported: ', reduce_func)

        self.update_network = MultilayerPerceptron(
            d_input=hidden_dim,
            d_hidden=(hidden_dim,) * (update_net_layers - 1),
            d_output=hidden_dim,
            batch_norm=True,
            batch_norm_momentum=batch_norm_momentum)

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        graph.update_all(message_func=self.message_function,
                         reduce_func=self.reduce_func(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)

    def message_function(self, edges):
        message_input = torch.cat(
            [edges.src['feat'], edges.dst['feat'], edges.data['d']], dim=-1)
        message = self.message_network(message_input)
        edges.data['d'] += message
        edge_weight = torch.sigmoid(self.soft_edge_network(message))
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        h = nodes.data['feat']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['feat']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'feat': output}


EPS = 1e-5


def aggregate_mean(h, **kwargs):
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h, **kwargs):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3, **kwargs):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    return torch.sum(h, dim=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output


def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d["log"] / np.log(D + 1))


PNA_AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": partial(aggregate_moment, n=3),
    "moment4": partial(aggregate_moment, n=4),
    "moment5": partial(aggregate_moment, n=5),
}

PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class PNA(nn.Module):
    """
    Message Passing Neural Network that does not use 3D information

    PNA: Principal Neighbourhood Aggregation 
        Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
        https://arxiv.org/abs/2004.05718
    """

    def __init__(self,
                 hidden_dim,
                 target_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 readout_aggregators: List[str],
                 readout_batchnorm: bool = True,
                 readout_hidden_dim=None,
                 readout_layers: int = 2,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 batch_norm_momentum=0.1,
                 **kwargs):
        super(PNA, self).__init__()
        self.node_gnn = PNAGNN(hidden_dim=hidden_dim,
                               aggregators=aggregators,
                               scalers=scalers,
                               residual=residual,
                               pairwise_distances=pairwise_distances,
                               activation=activation,
                               last_activation=last_activation,
                               mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm,
                               propagation_depth=propagation_depth,
                               dropout=dropout,
                               posttrans_layers=posttrans_layers,
                               pretrans_layers=pretrans_layers,
                               batch_norm_momentum=batch_norm_momentum)
        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators

        self.output = MultilayerPerceptron(
            d_input=hidden_dim * len(self.readout_aggregators),
            d_hidden=(readout_hidden_dim,) * (readout_layers - 1),
            d_output=target_dim,
            batch_norm=readout_batchnorm,
            batch_norm_momentum=batch_norm_momentum)

    def forward(self, graph: dgl.DGLGraph):
        self.node_gnn(graph)
        readouts_to_cat = [
            dgl.readout_nodes(graph, 'feat', op=aggr)
            for aggr in self.readout_aggregators
        ]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return self.output(readout)


class PNAGNN(nn.Module):

    def __init__(self,
                 hidden_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 batch_norm_momentum=0.1,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 **kwargs):
        super(PNAGNN, self).__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(in_dim=hidden_dim,
                         out_dim=int(hidden_dim),
                         in_dim_edges=hidden_dim,
                         aggregators=aggregators,
                         scalers=scalers,
                         pairwise_distances=pairwise_distances,
                         residual=residual,
                         dropout=dropout,
                         activation=activation,
                         last_activation=last_activation,
                         mid_batch_norm=mid_batch_norm,
                         last_batch_norm=last_batch_norm,
                         avg_d={"log": 1.0},
                         posttrans_layers=posttrans_layers,
                         pretrans_layers=pretrans_layers,
                         batch_norm_momentum=batch_norm_momentum),)
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, graph: dgl.DGLGraph):
        graph.ndata['feat'] = self.atom_encoder(graph.ndata['x'])
        graph.edata['feat'] = self.bond_encoder(graph.edata['edge_attr'])

        for mp_layer in self.mp_layers:
            mp_layer(graph)


class PNALayer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        aggregators: List[str],
        scalers: List[str],
        activation: Union[Callable, str] = "relu",
        last_activation: Union[Callable, str] = "none",
        dropout: float = 0.0,
        residual: bool = True,
        pairwise_distances: bool = False,
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum=0.1,
        avg_d: Dict[str, float] = {"log": 1.0},
        posttrans_layers: int = 2,
        pretrans_layers: int = 1,
    ):
        super(PNALayer, self).__init__()
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MultilayerPerceptron(
            d_input=(2 * in_dim + in_dim_edges +
                     1) if self.pairwise_distances else
            (2 * in_dim + in_dim_edges),
            d_output=in_dim,
            d_hidden=(in_dim,) * (pretrans_layers - 1),
            batch_norm=True,
            batch_norm_momentum=batch_norm_momentum,
            dropout=dropout)

        self.posttrans = MultilayerPerceptron(
            d_input=(len(self.aggregators) * len(self.scalers) + 1) * in_dim,
            d_hidden=(out_dim,) * (posttrans_layers - 1),
            d_output=out_dim,
            batch_norm=True,
            batch_norm_momentum=batch_norm_momentum,
            dropout=dropout)

    def forward(self, g):
        h = g.ndata['feat']
        h_in = h
        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['feat']], dim=-1)
        # post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in

        g.ndata['feat'] = h

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        The message function to generate messages along the edges.
        """
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data['feat']
        h = nodes.mailbox["e"]
        D = h.shape[-2]
        h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        h = torch.cat(h_to_cat, dim=-1)

        if len(self.scalers) > 1:
            h = torch.cat(
                [scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers],
                dim=-1)

        return {'feat': h}

    def pretrans_edges(self, edges) -> Dict[str, torch.Tensor]:
        r"""
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).
        """
        if self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x'])**2,
                                         dim=-1)[:, None]
            z2 = torch.cat([
                edges.src['feat'], edges.dst['feat'], edges.data['feat'],
                squared_distance
            ],
                           dim=-1)
        elif not self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src['x'] - edges.dst['x'])**2,
                                         dim=-1)[:, None]
            z2 = torch.cat(
                [edges.src['feat'], edges.dst['feat'], squared_distance],
                dim=-1)
        elif self.edge_features and not self.pairwise_distances:
            z2 = torch.cat(
                [edges.src['feat'], edges.dst['feat'], edges.data['feat']],
                dim=-1)
        else:
            z2 = torch.cat([edges.src['feat'], edges.dst['feat']], dim=-1)
        return {"e": self.pretrans(z2)}


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2**torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze()