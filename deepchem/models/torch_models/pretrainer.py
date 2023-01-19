import deepchem as dc
from deepchem.models import TorchModel
from deepchem.models.losses import L2Loss, L1Loss
import torch
import torch.nn as nn
import numpy as np
from deepchem.utils.typing import ArrayLike, LossFn, OneOrMany
from deepchem.data import Dataset, NumpyDataset
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os

class ToyModel(nn.Module):
    def __init__(self, input_dim, d_hidden, d_output):
        super(ToyModel, self).__init__()
        self._embedding = nn.Linear(input_dim, d_hidden)
        self._head = nn.Linear(d_hidden, d_output)
        self.model = nn.Sequential(self._embedding, self._head) 
    def forward(self, x):
        return self.model(x)
    
class ToyTorchModel(TorchModel):
    def __init__(self, input_dim, d_hidden, d_output, **kwargs): 
        self.model = ToyModel(input_dim, d_hidden, d_output)
        loss = L2Loss()
        super(ToyTorchModel, self).__init__(self.model, loss=loss, **kwargs) 
    def get_head(self):
        return self.model._head
    def get_embedding(self):
        return self.model._embedding
        

class Pretrainer(TorchModel): 
    def __init__(self,
                 torchmodel:TorchModel,
                 **kwargs): 
        self.torchmodel = torchmodel
        super().__init__(torchmodel.model, torchmodel.loss, model_dir=torchmodel.model_dir,**kwargs)
    def freeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = False
    def unfreeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = True
    def _define_pretrain_loss(self):
        return NotImplementedError("Subclass must define the pretrain loss")
    def _swap_head(self): 
        return NotImplementedError("Subclass must define how to change the head")
    def fit(self,
          dataset: Dataset,
          nb_epoch: int = 10,
          max_checkpoints_to_keep: int = 5,
          checkpoint_interval: int = 1000,
          deterministic: bool = False,
          restore: bool = False,
          variables: Optional[List[torch.nn.Parameter]] = None,
          loss: Optional[LossFn] = None,
          callbacks: Union[Callable, List[Callable]] = [],
          all_losses: Optional[List[float]] = None) -> float:
        
        """Fit the model to the data in dataset.
        
        Checkpoints saved at every interval will have the pretraining head of the model, not the original head.
        """
        self._swap_head()
        super().fit(dataset, nb_epoch, max_checkpoints_to_keep, checkpoint_interval, deterministic, restore, variables, loss, callbacks, all_losses) #use assignment map to save only embedding layers
        self._swap_head()
        self.save_checkpoint(model_dir = os.path.join(self.model_dir, 'pretrained'))  # model_dir=os.path.join(self.model_dir, 'completed_pretrained')


class ToyPretrainer(Pretrainer):
    def __init__(self, 
                 torchmodel:ToyTorchModel,
                 **kwargs): 
        torchmodel.loss = self._define_pretrain_loss()
        self.torchmodel = torchmodel
        self.embedding_dim = torchmodel.model._embedding.out_features
        self.old_head = torchmodel.model._head
        self.new_head = self._generate_head()
        super().__init__(torchmodel, **kwargs)
    def _generate_head(self):
        return nn.Linear(self.embedding_dim, 2)
    def _swap_head(self):
        self.torchmodel.model._head = self.new_head
        self.new_head = self.old_head
        self.old_head = self.new_head
    def _define_pretrain_loss(self):
        return L1Loss()

        
n_samples = 10
input_size = 15
hidden_layers = 3
n_tasks = 3

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
ft_dataset = dc.data.NumpyDataset(X, y)

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
pt_dataset = dc.data.NumpyDataset(X, y)

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
test_dataset = dc.data.NumpyDataset(X)


toy = ToyTorchModel(input_size, hidden_layers, n_tasks)
toy2 = ToyTorchModel(input_size, hidden_layers, n_tasks, model_dir = './testfolder')
###

pretrainer = ToyPretrainer(toy2) 

# pretrainer.freeze_embedding()
pretrainer.fit(pt_dataset, nb_epoch=100) # will save with pretrained head at every interval
# pretrainer.save_checkpoint(model_dir = os.path.join(pretrainer.model_dir, 'completed_pretrained')) # will save with original head

### load_from_pretrained()

# toy.load_from_pretrained() # does not work on master
# toy.load_from_pretrained(toy) # does not work on master, no ckpt found (self.model_dir not passed)
# toy.load_from_pretrained(toy, checkpoint='./testfolder/checkpoint5.pt') # works on master
# toy.load_from_pretrained(toy2, include_top=False) # works on master

# toy.load_from_pretrained() # works on pretrainer branch. uses self.model as source_model

### restore()

# .restore() load from model_dir does not work as expected, need to pass model_dir again
# toy.restore() # works on master
# toy.restore(checkpoint='/home/tony/github/bio/deepchem/deepchem/models/torch_models/testfolder/checkpoint5.pt') # works on master
# toy.restore(model_dir='/home/tony/github/bio/deepchem/deepchem/models/torch_models/testfolder') # works on master
# toy.restore() # works on pretrainer 


#### only load embedding, if defined
# default behavior, load embedding flag. if not expects assignment mapping
# load embedding only as option in load_from_pretrained
# toy.load_from_pretrained(pretrained_model_dir='/home/tony/github/bio/deepchem/deepchem/models/torch_models/testfolder', load_embedding=True) 


###

# toy.fit(ft_dataset, nb_epoch=100, checkpoint_interval=10) 
# toy.save_checkpoint(model_dir='/home/tony/github/bio/deepchem/deepchem/models/torch_models/testfolder/')

# preds = toy.predict(test_dataset)
# print(preds)