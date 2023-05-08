import pandas as pd
from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
from deepchem.models.torch_models.infomax3d import PNA, Net3D
import torch
from deepchem.feat.graph_data import BatchGraphData

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
model3d = Net3D(node_dim=75,
                edge_dim=12,
                hidden_dim=64,
                target_dim=5,
                readout_aggregators=["sum"])

featurizer = RDKitConformerFeaturizer(num_conformers=2)

graphs = featurizer.featurize(datapoints)
criterion = torch.nn.MSELoss()
total_loss = 0
for graph in graphs:
    for conf in graph:
        # graph = BatchGraphData(graph)
        view2d = model2d(conf.to_dgl_graph())
        view3d = model3d(conf.to_dgl_graph())
        loss = criterion(view2d, view3d)
        total_loss += loss

print(total_loss)