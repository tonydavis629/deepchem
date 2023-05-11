from functools import partial
from math import sqrt
from typing import Callable, Dict, List, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deepchem.feat.molecule_featurizers.conformer_featurizer import (
    full_atom_feature_dims,
    full_bond_feature_dims,
)
from deepchem.models.torch_models.layers import MultilayerPerceptron


class AtomEncoder(torch.nn.Module):
    """
    Encodes atom features into embeddings.

    Parameters
    ----------
    emb_dim : int
        The dimension that the returned embedding will have.
    padding : bool, optional (default=False)
        If true then the last index will be used for padding.

    Examples
    --------
    >>> atom_encoder = AtomEncoder(emb_dim=32)
    >>> atom_features = torch.tensor([[1, 6, 0], [2, 7, 1]])
    >>> atom_embeddings = atom_encoder(atom_features)
    """

    def __init__(self, emb_dim, padding=False):
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
    """
    Encodes bond features into embeddings.

    Parameters
    ----------
    emb_dim : int
        The dimension that the returned embedding will have.
    padding : bool, optional (default=False)
        If true then the last index will be used for padding.

    Examples
    --------
    >>> bond_encoder = BondEncoder(emb_dim=32)
    >>> bond_features = torch.tensor([[1, 0], [2, 1]])
    >>> bond_embeddings = bond_encoder(bond_features)
    """

    def __init__(self, emb_dim, padding=False):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for dim in full_bond_feature_dims:
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


class Net3D(nn.Module):
    """
    Net3D is a 3D graph neural network that expects a DGL graph input with 3D coordiantes.

    Parameters
    ----------
    hidden_dim : int
        The dimension of the hidden layers.
    target_dim : int
        The dimension of the output layer.
    readout_aggregators : List[str]
        A list of aggregator functions for the readout layer.
    batch_norm : bool, optional (default=False)
        Whether to use batch normalization.
    node_wise_output_layers : int, optional (default=2)
        The number of output layers for each node.
    readout_batchnorm : bool, optional (default=True)
        Whether to use batch normalization in the readout layer.
    batch_norm_momentum : float, optional (default=0.1)
        The momentum for the batch normalization layers.
    reduce_func : str, optional (default='sum')
        The reduce function to use for aggregating messages.
    dropout : float, optional (default=0.0)
        The dropout rate for the layers.
    propagation_depth : int, optional (default=4)
        The number of propagation layers in the network.
    readout_layers : int, optional (default=2)
        The number of readout layers in the network.
    readout_hidden_dim : int, optional (default=None)
        The dimension of the hidden layers in the readout network.
    fourier_encodings : int, optional (default=0)
        The number of Fourier encodings to use.
    activation : str, optional (default='SiLU')
        The activation function to use in the network.
    update_net_layers : int, optional (default=2)
        The number of update network layers.
    message_net_layers : int, optional (default=2)
        The number of message network layers.
    use_node_features : bool, optional (default=False)
        Whether to use node features as input.

    Examples
    --------
    >>> net3d = Net3D(hidden_dim=32, target_dim=1, readout_aggregators=['mean'])
    >>> graph = dgl.DGLGraph()
    >>> output = net3d(graph)
    """

    def __init__(self,
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
                 use_node_features=False):
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
        return {'d': F.silu(self.edge_input(edges.data['edge_attr']))}


class Net3DLayer(nn.Module):
    """
    Net3DLayer is a single layer of a 3D graph neural network.

    Parameters
    ----------
    edge_dim : int
        The dimension of the edge features.
    reduce_func : str
        The reduce function to use for aggregating messages.
    hidden_dim : int
        The dimension of the hidden layers.
    batch_norm : bool, optional (default=False)
        Whether to use batch normalization.
    batch_norm_momentum : float, optional (default=0.1)
        The momentum for the batch normalization layers.
    dropout : float, optional (default=0.0)
        The dropout rate for the layers.
    mid_activation : str, optional (default='SiLU')
        The activation function to use in the network.
    message_net_layers : int, optional (default=2)
        The number of message network layers.
    update_net_layers : int, optional (default=2)
        The number of update network layers.

    Examples
    --------
    >>> net3d_layer = Net3DLayer(edge_dim=32, reduce_func='sum', hidden_dim=32)
    >>> graph = dgl.DGLGraph()
    >>> net3d_layer(graph)
    """

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
    """
    Compute the mean of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Mean of the input tensor along the second to last dimension.
    """
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    """
    Compute the max of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Max of the input tensor along the second to last dimension.
    """
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    """
    Compute the min of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    torch.Tensor
        Min of the input tensor along the second to last dimension.
    """
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    """
    Compute the standard deviation of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Standard deviation of the input tensor along the second to last dimension.
    """
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h, **kwargs):
    """
    Compute the variance of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Variance of the input tensor along the second to last dimension.
    """
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3, **kwargs):
    """
    Compute the nth moment of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    n : int, optional, default=3
        The order of the moment to compute.

    Returns
    -------
    torch.Tensor
        Nth moment of the input tensor along the second to last dimension.
    """
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    """
    Compute the sum of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Sum of the input tensor along the second to last dimension.
    """
    return torch.sum(h, dim=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output
def scale_identity(h, D=None, avg_d=None):
    """
    Identity scaling function.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    D : torch.Tensor, optional
        Degree tensor.
    avg_d : dict, optional
        Dictionary containing averages over the training set.

    Returns
    -------
    torch.Tensor
        Scaled input tensor.
    """
    return h


def scale_amplification(h, D, avg_d):
    """
    Amplification scaling function. log(D + 1) / d * h where d is the average of the ``log(D + 1)`` in the training set

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    D : torch.Tensor
        Degree tensor.
    avg_d : dict
        Dictionary containing averages over the training set.

    Returns
    -------
    torch.Tensor
        Scaled input tensor.
    """
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    """
    Attenuation scaling function. (log(D + 1))^-1 / d * X where d is the average of the ``log(D + 1))^-1`` in the training set

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    D : torch.Tensor
        Degree tensor.
    avg_d : dict
        Dictionary containing averages over the training set.

    Returns
    -------
    torch.Tensor
        Scaled input tensor.
    """
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
    Principal Neighbourhood Aggregation Message Passing Neural Network [1]. This is a 2D GNN.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layers.
    target_dim : int
        Dimension of the output layer.
    aggregators : List[str]
        List of aggregator functions to use.
    scalers : List[str]
        List of scaler functions to use.
    readout_aggregators : List[str]
        List of readout aggregator functions to use.
    readout_batchnorm : bool, optional, default=True
        Whether to use batch normalization in the readout layers.
    readout_hidden_dim : int, optional
        Dimension of the hidden layers in the readout layers. If not provided, it will be set to the value of `hidden_dim`.
    readout_layers : int, optional, default=2
        Number of layers in the readout layers.
    residual : bool, optional, default=True
        Whether to use residual connections.
    pairwise_distances : bool, optional, default=False
        Whether to use pairwise distances.
    activation : Union[Callable, str], optional, default="relu"
        Activation function to use.
    last_activation : Union[Callable, str], optional, default="none"
        Last activation function to use.
    mid_batch_norm : bool, optional, default=False
        Whether to use batch normalization in the middle layers.
    last_batch_norm : bool, optional, default=False
        Whether to use batch normalization in the last layer.
    propagation_depth : int, optional, default=5
        Number of propagation layers.
    dropout : float, optional, default=0.0
        Dropout rate.
    posttrans_layers : int, optional, default=1
        Number of post-transformation layers.
    pretrans_layers : int, optional, default=1
        Number of pre-transformation layers.
    batch_norm_momentum : float, optional, default=0.1
        Momentum for the batch normalization layers.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from deepchem.models.torch_models.infomax3d import PNA
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    >>> g.ndata['x'] = torch.randn(3, 3)
    >>> g.edata['edge_attr'] = torch.randn(3, 3)
    >>> model = PNA(hidden_dim=16, target_dim=1, aggregators=['mean', 'sum'], scalers=['identity'], readout_aggregators=['mean'])
    >>> output = model(g)

    References
    ----------
    .. [1] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
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
    """
    Principal Neighbourhood Aggregation Graph Neural Network

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layers.
    aggregators : List[str]
        List of aggregator functions to use.
    scalers : List[str]
        List of scaler functions to use.
    residual : bool, optional, default=True
        Whether to use residual connections.
    pairwise_distances : bool, optional, default=False
        Whether to use pairwise distances.
    activation : Union[Callable, str], optional, default="relu"
        Activation function to use.
    last_activation : Union[Callable, str], optional, default="none"
        Last activation function to use.
    mid_batch_norm : bool, optional, default=False
        Whether to use batch normalization in the middle layers.
    last_batch_norm : bool, optional, default=False
        Whether to use batch normalization in the last layer.
    batch_norm_momentum : float, optional, default=0.1
        Momentum for the batch normalization layers.
    propagation_depth : int, optional, default=5
        Number of propagation layers.
    dropout : float, optional, default=0.0
        Dropout rate.
    posttrans_layers : int, optional, default=1
        Number of post-transformation layers.
    pretrans_layers : int, optional, default=1
        Number of pre-transformation layers.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from deepchem.models.torch_models.infomax3d import PNAGNN
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    >>> g.ndata['x'] = torch.randn(3, 3)
    >>> g.edata['edge_attr'] = torch.randn(3, 3)
    >>> model = PNAGNN(hidden_dim=16, aggregators=['mean', 'sum'], scalers=['identity'])
    >>> model(g)
    """

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
    """
    Principal Neighbourhood Aggregation Layer.

    Parameters
    ----------
    in_dim : int
        Input dimension of the node features.
    out_dim : int
        Output dimension of the node features.
    in_dim_edges : int
        Input dimension of the edge features.
    aggregators : List[str]
        List of aggregator functions to use.
    scalers : List[str]
        List of scaler functions to use.
    activation : Union[Callable, str], optional, default="relu"
        Activation function to use.
    last_activation : Union[Callable, str], optional, default="none"
        Last activation function to use.
    dropout : float, optional, default=0.0
        Dropout rate.
    residual : bool, optional, default=True
        Whether to use residual connections.
    pairwise_distances : bool, optional, default=False
        Whether to use pairwise distances.
    mid_batch_norm : bool, optional, default=False
        Whether to use batch normalization in the middle layers.
    last_batch_norm : bool, optional, default=False
        Whether to use batch normalization in the last layer.
    batch_norm_momentum : float, optional, default=0.1
        Momentum for the batch normalization layers.
    avg_d : Dict[str, float], optional, default={"log": 1.0}
        Dictionary containing the average degree of the graph.
    posttrans_layers : int, optional, default=2
        Number of post-transformation layers.
    pretrans_layers : int, optional, default=1
        Number of pre-transformation layers.
    """
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
        """
        Forward pass of the PNA layer.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph.
        """
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
        """
        The message function to generate messages along the edges.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Batch of edges.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the edge features.
        """
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        """
        The reduce function to aggregate the messages.
        Apply the aggregators and scalers, and concatenate the results.

        Parameters
        ----------
        nodes : dgl.NodeBatch
            Batch of nodes.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the aggregated node features.
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
        """
        Return a mapping to the concatenation of the features from
        the source node, the destination node, and the edge between them (if applicable).

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Batch of edges.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the concatenated features.
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
    """
    Fourier encode the input tensor `x` based on the specified number of encodings.

    This function applies a Fourier encoding to the input tensor `x` by dividing
    it by a range of scales (2^i for i in range(num_encodings)) and then
    concatenating the sine and cosine of the scaled values. Optionally, the
    original input tensor can be included in the output.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be Fourier encoded.
    num_encodings : int, optional, default=4
        Number of Fourier encodings to apply.
    include_self : bool, optional, default=True
        Whether to include the original input tensor in the output.

    Returns
    -------
    torch.Tensor
        Fourier encoded tensor.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> encoded_x = fourier_encode_dist(x, num_encodings=4, include_self=True)
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2**torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze()
