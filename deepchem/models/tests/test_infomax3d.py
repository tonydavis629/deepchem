import pandas as pd
from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
from deepchem.models.torch_models.infomax3d import PNA, Net3D
import torch
import dgl
import numpy as np
# from deepchem.feat.graph_data import BatchGraphData

example_data = pd.read_csv(
    '/home/tony/github/deepchem/deepchem/models/tests/assets/example_regression.csv'
)

# return list of SMILES strings
ids = example_data['Compound ID'].tolist()
datapoints = example_data['smiles'].tolist()
targets = example_data['outcome'].tolist()

model2d = PNA(in_dim=75,
              target_dim=5,
              hidden_dim=64,
              aggregators=['sum', 'mean', 'max'],
              scalers=["identity"],
              readout_aggregators=["sum"])
model3d = Net3D(hidden_dim=64,
                target_dim=5,
                readout_aggregators=["sum"])

featurizer = RDKitConformerFeaturizer(num_conformers=2)
torch.manual_seed(0)
np.random.seed(0)
graphs = featurizer.featurize(datapoints)  # returns a list of lists of conformers
criterion = torch.nn.MSELoss()
total_loss = 0
for graph in graphs:
    confs = [conf.to_dgl_graph() for conf in graph]
    batch = dgl.batch(confs)
    # graph = BatchGraphData(graph)
    view2d = model2d(batch)
    view3d = model3d(batch)
    loss = criterion(view2d,
                     view3d)
    total_loss += loss

print(total_loss)