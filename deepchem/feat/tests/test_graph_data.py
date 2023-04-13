import unittest

import numpy as np
import pytest

from deepchem.feat.graph_data import BatchGraphData, GraphData


class TestGraph(unittest.TestCase):

    @pytest.mark.torch
    def test_graph_data(self):
        num_nodes, num_node_features = 5, 32
        num_edges, num_edge_features = 6, 32
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_features = np.random.random_sample((num_edges, num_edge_features))
        edge_index = np.array([
            [0, 1, 2, 2, 3, 4],
            [1, 2, 0, 3, 4, 0],
        ])
        node_pos_features = None
        # z is kwargs
        z = np.random.random(5)

        graph = GraphData(node_features=node_features,
                          edge_index=edge_index,
                          edge_features=edge_features,
                          node_pos_features=node_pos_features,
                          z=z)

        assert graph.num_nodes == num_nodes
        assert graph.num_node_features == num_node_features
        assert graph.num_edges == num_edges
        assert graph.num_edge_features == num_edge_features
        assert graph.z.shape == z.shape
        assert str(
            graph
        ) == 'GraphData(node_features=[5, 32], edge_index=[2, 6], edge_features=[6, 32], z=[5])'

        # check convert function
        pyg_graph = graph.to_pyg_graph()
        from torch_geometric.data import Data
        assert isinstance(pyg_graph, Data)
        assert tuple(pyg_graph.z.shape) == z.shape

        dgl_graph = graph.to_dgl_graph()
        from dgl import DGLGraph
        assert isinstance(dgl_graph, DGLGraph)

    @pytest.mark.torch
    def test_invalid_graph_data(self):
        with self.assertRaises(ValueError):
            invalid_node_features_type = list(np.random.random_sample((5, 32)))
            edge_index = np.array([
                [0, 1, 2, 2, 3, 4],
                [1, 2, 0, 3, 4, 0],
            ])
            _ = GraphData(
                node_features=invalid_node_features_type,
                edge_index=edge_index,
            )

        with self.assertRaises(ValueError):
            node_features = np.random.random_sample((5, 32))
            invalid_edge_index_shape = np.array([
                [0, 1, 2, 2, 3, 4],
                [1, 2, 0, 3, 4, 5],
            ])
            _ = GraphData(
                node_features=node_features,
                edge_index=invalid_edge_index_shape,
            )

        with self.assertRaises(ValueError):
            node_features = np.random.random_sample((5, 5))
            invalid_edge_index_shape = np.array([
                [0, 1, 2, 2, 3, 4],
                [1, 2, 0, 3, 4, 0],
                [2, 2, 1, 4, 0, 3],
            ],)
            _ = GraphData(
                node_features=node_features,
                edge_index=invalid_edge_index_shape,
            )

        with self.assertRaises(TypeError):
            node_features = np.random.random_sample((5, 32))
            _ = GraphData(node_features=node_features)

    @pytest.mark.torch
    def test_batch_graph_data(self):
        num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
        num_node_features, num_edge_features = 32, 32
        edge_index_list = [
            np.array([[0, 1], [1, 2]]),
            np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
            np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
        ]

        graph_list = [
            GraphData(node_features=np.random.random_sample(
                (num_nodes_list[i], num_node_features)),
                      edge_index=edge_index_list[i],
                      edge_features=np.random.random_sample(
                          (num_edge_list[i], num_edge_features)),
                      node_pos_features=None) for i in range(len(num_edge_list))
        ]
        batch = BatchGraphData(graph_list)

        assert batch.num_nodes == sum(num_nodes_list)
        assert batch.num_node_features == num_node_features
        assert batch.num_edges == sum(num_edge_list)
        assert batch.num_edge_features == num_edge_features
        assert batch.graph_index.shape == (sum(num_nodes_list),)
        assert batch.edge_index.max() == sum(num_edge_list)
        assert batch.edge_index.shape == (2, sum(num_edge_list))

    @pytest.mark.torch
    def test_graph_data_single_atom_mol(self):
        """
        Test for graph data when no edges in the graph (example: single atom mol)
        """
        num_nodes, num_node_features = 1, 32
        num_edges = 0
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_index = np.empty((2, 0), dtype=int)

        graph = GraphData(node_features=node_features, edge_index=edge_index)

        assert graph.num_nodes == num_nodes
        assert graph.num_node_features == num_node_features
        assert graph.num_edges == num_edges
        assert str(
            graph
        ) == 'GraphData(node_features=[1, 32], edge_index=[2, 0], edge_features=None)'

    @pytest.mark.torch
    def test_graphdata_numpy_to_torch(self):
        """
        Test for converting GraphData numpy arrays to torch tensors
        """
        import torch
        num_nodes, num_node_features = 5, 32
        num_edges, num_edge_features = 6, 32
        node_features = np.random.random_sample((num_nodes, num_node_features))
        edge_features = np.random.random_sample((num_edges, num_edge_features))
        edge_index = np.array([
            [0, 1, 2, 2, 3, 4],
            [1, 2, 0, 3, 4, 0],
        ])
        node_pos_features = None
        # z is kwargs
        z = np.random.random(5)

        graph_np = GraphData(node_features=node_features,
                             edge_index=edge_index,
                             edge_features=edge_features,
                             node_pos_features=node_pos_features,
                             z=z)
        graph = graph_np.numpy_to_torch()

        assert isinstance(graph.node_features, torch.Tensor)
        assert isinstance(graph.edge_index, torch.Tensor)
        assert isinstance(graph.edge_features, torch.Tensor)
        assert graph.node_pos_features is None
        assert isinstance(graph.z, torch.Tensor)

    @pytest.mark.torch
    def test_batchgraphdata_numpy_to_torch(self):
        import torch
        num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
        num_node_features, num_edge_features = 32, 32
        edge_index_list = [
            np.array([[0, 1], [1, 2]]),
            np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
            np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
        ]

        graph_list = [
            GraphData(node_features=np.random.random_sample(
                (num_nodes_list[i], num_node_features)),
                      edge_index=edge_index_list[i],
                      edge_features=np.random.random_sample(
                          (num_edge_list[i], num_edge_features)),
                      node_pos_features=None) for i in range(len(num_edge_list))
        ]
        batched_graph = BatchGraphData(graph_list)

        batched_graph = batched_graph.numpy_to_torch()

        assert isinstance(batched_graph, BatchGraphData)
        assert isinstance(batched_graph.node_features, torch.Tensor)
        assert isinstance(batched_graph.edge_index, torch.Tensor)
        assert isinstance(batched_graph.edge_features, torch.Tensor)
        assert batched_graph.node_pos_features is None

    def test_batch_graph_data_with_user_defined_attributes(self):
        edge_index = np.array([[0, 1], [1, 0]])
        node_features_shape = 5
        n_nodes = 2
        g1 = GraphData(node_features=np.random.randn(n_nodes,
                                                     node_features_shape),
                       edge_index=edge_index,
                       user_defined_attribute1=[0, 1])

        g2 = GraphData(node_features=np.random.randn(n_nodes,
                                                     node_features_shape),
                       edge_index=edge_index,
                       user_defined_attribute1=[2, 3])

        g3 = GraphData(node_features=np.random.randn(n_nodes,
                                                     node_features_shape),
                       edge_index=edge_index,
                       user_defined_attribute1=[4, 5])
        g = BatchGraphData([g1, g2, g3])

        assert hasattr(g, 'user_defined_attribute1')
        assert (g.user_defined_attribute1 == np.array([[0, 1], [2, 3],
                                                       [4, 5]])).all()
