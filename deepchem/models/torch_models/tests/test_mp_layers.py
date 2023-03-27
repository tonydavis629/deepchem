import unittest
try:
    import torch
except:
    pass


class TestGINConv(unittest.TestCase):

    def setUp(self):
        from deepchem.models.torch_models.mp_layers import GINConv
        self.emb_dim = 16
        self.aggr = "add"
        self.ginconv = GINConv(self.emb_dim, self.aggr)

    def test_forward(self):
        x = torch.randn(5, self.emb_dim)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                  dtype=torch.long)
        edge_attr = torch.tensor([[1, 0], [2, 1], [3, 0], [4, 1], [1, 0]],
                                 dtype=torch.long)

        out = self.ginconv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, self.emb_dim))

    def test_message(self):
        x_j = torch.randn(5, self.emb_dim)
        edge_attr = torch.randn(5, self.emb_dim)

        message_out = self.ginconv.message(x_j, edge_attr)
        self.assertEqual(message_out.shape, (5, self.emb_dim))

    def test_update(self):
        aggr_out = torch.randn(5, self.emb_dim)

        update_out = self.ginconv.update(aggr_out)
        self.assertEqual(update_out.shape, (5, self.emb_dim))


class TestGCNConv(unittest.TestCase):

    def setUp(self):
        from deepchem.models.torch_models.mp_layers import GCNConv
        self.emb_dim = 16
        self.aggr = "add"
        self.gcnconv = GCNConv(self.emb_dim, self.aggr)

    def test_norm(self):
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                  dtype=torch.long)
        num_nodes = 5
        dtype = torch.float32

        norm = self.gcnconv.norm(edge_index, num_nodes, dtype)
        self.assertEqual(norm.shape, (edge_index.shape[1],))

    def test_forward(self):
        x = torch.randn(5, self.emb_dim)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                  dtype=torch.long)
        edge_attr = torch.tensor([[1, 0], [2, 1], [3, 0], [4, 1], [1, 0]],
                                 dtype=torch.long)

        out = self.gcnconv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, self.emb_dim))

    def test_message(self):
        x_j = torch.randn(5, self.emb_dim)
        edge_attr = torch.randn(5, self.emb_dim)
        norm = torch.ones(5)

        message_out = self.gcnconv.message(x_j, edge_attr, norm)
        self.assertEqual(message_out.shape, (5, self.emb_dim))


class TestGATConv(unittest.TestCase):

    def setUp(self):
        from deepchem.models.torch_models.mp_layers import GATConv
        self.emb_dim = 16
        self.heads = 2
        self.negative_slope = 0.2
        self.aggr = "add"
        self.gatconv = GATConv(self.emb_dim, self.heads, self.negative_slope,
                               self.aggr)

    def test_forward(self):
        x = torch.randn(5, self.emb_dim)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                  dtype=torch.long)
        edge_attr = torch.tensor([[1, 0], [2, 1], [3, 0], [4, 1], [1, 0]],
                                 dtype=torch.long)

        out = self.gatconv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, self.emb_dim))

    def test_message(self):
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                  dtype=torch.long)
        x_i = torch.randn(5, self.heads, self.emb_dim)
        x_j = torch.randn(5, self.heads, self.emb_dim)
        edge_attr = torch.randn(5, self.heads * self.emb_dim)

        message_out = self.gatconv.message(edge_index, x_i, x_j, edge_attr)
        self.assertEqual(message_out.shape, (5, self.heads, self.emb_dim))

    def test_update(self):
        aggr_out = torch.randn(5, self.heads, self.emb_dim)

        update_out = self.gatconv.update(aggr_out)
        self.assertEqual(update_out.shape, (5, self.emb_dim))


class TestGraphSAGEConv(unittest.TestCase):

    def setUp(self):
        from deepchem.models.torch_models.mp_layers import GraphSAGEConv
        self.emb_dim = 16
        self.aggr = "mean"
        self.graphsageconv = GraphSAGEConv(self.emb_dim, self.aggr)

    def test_forward(self):
        x = torch.randn(5, self.emb_dim)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                  dtype=torch.long)
        edge_attr = torch.tensor([[1, 0], [2, 1], [3, 0], [4, 1], [1, 0]],
                                 dtype=torch.long)

        out = self.graphsageconv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, self.emb_dim))

    def test_message(self):
        x_j = torch.randn(5, self.emb_dim)
        edge_attr = torch.randn(5, self.emb_dim)

        message_out = self.graphsageconv.message(x_j, edge_attr)
        self.assertEqual(message_out.shape, (5, self.emb_dim))

    def test_update(self):
        aggr_out = torch.randn(5, self.emb_dim)

        update_out = self.graphsageconv.update(aggr_out)
        self.assertEqual(update_out.shape, (5, self.emb_dim))


testgcn = TestGCNConv()
testgraphsage = TestGraphSAGEConv()
testgat = TestGATConv()
testgin = TestGINConv()

testgraphsage.setUp()
testgraphsage.test_forward()

testgin.setUp()
testgin.test_forward()

testgat.setUp()
testgat.test_forward()

testgcn.setUp()
testgcn.test_forward()