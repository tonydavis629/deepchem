import deepchem as dc
from deepchem.feat.molecule_featurizers.snap_featurizer import SNAPfeaturizer

import pytest
import os
# import torch
# from rdkit import Chem
# import numpy as np


def mol_to_graph_data_obj_simple(mol):


@pytest.mark.torch
def gen_dataset():
    # import torch
    # load sample dataset
    dir = os.path.dirname(os.path.abspath(__file__))
    # input_file = os.path.join(dir, '../../assets/example_classification.csv')
    assets_dir = os.path.abspath(os.path.join(dir, "../../tests/assets"))
    path = os.path.join(assets_dir, "example_classification.csv")
    # path = 'deepchem/models/tests/assets/example_classification.csv'
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=SNAPfeaturizer())
    # need to implement simple GraphFeaturizer to take in smiles and output GraphData
    dataset = loader.create_dataset(path)
    # dataset = [mol_to_graph_data_obj_simple(mol) for mol in dataset.X]
    return dataset


@pytest.mark.torch
def test_GINConv():
    from deepchem.models.torch_models.mp_layers import GINConv
    layer = GINConv(emb_dim=10)
    dataset = gen_dataset()[0]
    output = layer(dataset.x, dataset.edge_index, dataset.edge_attr)
    print(output)
test_GINConv()

@pytest.mark.torch
def test_GCNConv():
    from deepchem.models.torch_models.mp_layers import GCNConv
    layer = GCNConv(emb_dim=10)


@pytest.mark.torch
def test_GATConv():
    from deepchem.models.torch_models.mp_layers import GATConv
    layer = GATConv(emb_dim=10)


@pytest.mark.torch
def test_GraphSAGEConv():
    from deepchem.models.torch_models.mp_layers import GraphSAGEConv
    layer = GraphSAGEConv(emb_dim=10)
