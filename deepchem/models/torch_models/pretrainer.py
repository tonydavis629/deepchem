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

class ToyTorchModel(TorchModel):
    def __init__(self, 
                 input_dim, 
                 d_hidden, 
                 d_output, 
                 **kwargs): 
        self.loss = L2Loss()
        self.head = self.build_head(d_hidden, d_output)
        self.embedding = self.build_embedding(input_dim, d_hidden)
        self.model = self.build_model(self.embedding, self.head)
        super().__init__(self.model, self.loss, **kwargs) 
    def build_embedding(self, input_dim, d_hidden):
        return nn.Linear(input_dim, d_hidden)
    def build_head(self, d_hidden, d_output):
        return nn.Linear(d_hidden, d_output)
    def build_model(self,embedding,head):
        return nn.Sequential(embedding, head)

class Pretrainer(TorchModel): 
    """does not modify internal state of model"""
    def __init__(self,
                 torchmodel:TorchModel,
                 model_dir:str,
                 **kwargs): 
        super().__init__(torchmodel.model, torchmodel.loss, model_dir=model_dir,**kwargs) 
    def freeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = False
    def unfreeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = True
    def _define_pretrain_loss(self):
        return NotImplementedError("Subclass must define the pretrain loss")
        
class ToyPretrainer(Pretrainer): 
    def __init__(self, 
                 model:ToyTorchModel,
                 pt_tasks:int,
                 **kwargs):
                 
        self.head = self.build_head(model.embedding.out_features, pt_tasks)
        
        self.build_embedding = model.build_embedding
        self.build_model = model.build_model
        
        self.embedding = self.build_embedding(model.embedding.in_features, model.embedding.out_features)
        self.model = self.build_model(self.embedding, self.head)
        self.loss = self._define_pretrain_loss()
        
        torchmodel = TorchModel(self.model, self.loss, **kwargs)
        
        super().__init__(torchmodel, **kwargs)
        
    def _define_pretrain_loss(self):
        return L1Loss()
    def build_head(self, d_hidden, pt_tasks): 
        return nn.Linear(d_hidden, pt_tasks)

np.random.seed(123)
n_samples = 10
input_size = 15
d_hidden = 2
n_tasks = 3
pt_tasks = 5

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, pt_tasks)).astype(np.float32)
pt_dataset = dc.data.NumpyDataset(X, y)

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
ft_dataset = dc.data.NumpyDataset(X, y)

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
test_dataset = dc.data.NumpyDataset(X)


toy = ToyTorchModel(input_size, d_hidden, n_tasks, model_dir = './testfolder1')
toy2 = ToyTorchModel(input_size, d_hidden, n_tasks, model_dir = './testfolder2')

pretrainer = ToyPretrainer(toy, pt_tasks=5, model_dir = './testfolder2') 
pretrainer.fit(pt_dataset, nb_epoch=100, checkpoint_interval=10) 

### build new model, fit normally, then load from pretrained with only embedding
toy2.load_from_pretrained(pretrainer, include_top=False, model_dir = './testfolder2') # works 
toy2.fit(ft_dataset, nb_epoch=100, checkpoint_interval=10)

toy.fit(ft_dataset, nb_epoch=100, checkpoint_interval=10)
### load_from_pretrained() test, works
preds = toy2.predict(test_dataset)
print('toy2 preds: \n', preds)
preds = toy.predict(test_dataset)
print('toy preds: \n', preds)