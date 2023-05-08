import pandas as pd
from deepchem.feat.molecule_featurizers import RDKitConformerFeaturizer

example_data = pd.read_csv(
    '/home/tony/github/deepchem/deepchem/models/tests/assets/example_regression.csv'
)

# return list of SMILES strings
ids = example_data['Compound ID'].tolist()
datapoints = example_data['smiles'].tolist()
targets = example_data['outcome'].tolist()

graphs = featurize(datapoints)
criterion = torch.nn.MSELoss()
total_loss = 0
for graph in graphs:
    view2d = model2d(graph.to_dgl_graph())
    view3d = model3d(graph.to_dgl_graph())
    loss = criterion(view2d, view3d)
    total_loss += loss

print(total_loss)