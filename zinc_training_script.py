import deepchem as dc
from deepchem.models.torch_models.infograph import InfoGraphModel
from deepchem.feat import MolGraphConvFeaturizer

save_dir = '/home/tony/github/data/zinc'

featurizer = MolGraphConvFeaturizer(use_edges=True)
tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer=featurizer,
                                                      splitter='random',
                                                      save_dir=save_dir)
train_dataset, valid_dataset, test_dataset = datasets
num_feat = 30  # max([train_dataset.X[i].num_node_features for i in range(len(train_dataset))])
num_edge = 11  # max([train_dataset.X[i].num_edge_features for i in range(len(train_dataset))])
dim = 64
use_unsup_loss = True
sep_encoder = False
model = InfoGraphModel(num_feat,
                       num_edge,
                       dim,
                       use_unsup_loss,
                       sep_encoder,
                       tensorboard=True,
                       model_dir=save_dir,
                       batch_size=800)
model.fit(train_dataset, nb_epoch=5, checkpoint_interval=1)
