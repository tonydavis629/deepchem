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
        embedding = nn.Linear(input_dim, d_hidden)
        head = nn.Linear(d_hidden, d_output)
        self.model = nn.Sequential(embedding, head) 
    def forward(self, x):
        return self.model(x)
    
class ToyTorchModel(TorchModel):
    def __init__(self, input_dim, d_hidden, d_output, **kwargs): 
        # self.model = ToyModel(input_dim, d_hidden, d_output)
        loss = L2Loss()
        self.embedding = self.build_embedding(input_dim, d_hidden)
        self.head = self.build_head(d_hidden, d_output)
        self.model = self.build_model(self.embedding,self.head)
        super(ToyTorchModel, self).__init__(self.model, loss=loss, **kwargs) 
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
                 **kwargs): 
        # self.torchmodel = torchmodel
        # self.model = torchmodel.model.get_embedding()
        # self.pretrain_model_dir = os.path.join(torchmodel.model_dir, "pretrain")
        super().__init__(torchmodel.model, torchmodel.loss, model_dir=torchmodel.model_dir,**kwargs) 
    def freeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = False
    def unfreeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = True
    def _define_pretrain_loss(self):
        return NotImplementedError("Subclass must define the pretrain loss")


class ToyPretrainer(Pretrainer):
    def __init__(self, 
                 torchmodel:ToyTorchModel,
                 d_output:int = 5,
                 **kwargs): 
        torchmodel.head = self.build_head(torchmodel.embedding.out_features, d_output)
        torchmodel.model = torchmodel.build_model(torchmodel.embedding,torchmodel.head)
        torchmodel.loss = self._define_pretrain_loss()
        super().__init__(torchmodel, **kwargs)
    def _define_pretrain_loss(self):
        return L1Loss()
    def build_head(self, d_hidden, d_output): #can overwrite build_head
        return nn.Linear(d_hidden, d_output)

np.random.seed(123)
n_samples = 10
input_size = 15
d_hidden = 2
n_tasks = 3

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
pt_dataset = dc.data.NumpyDataset(X, y)

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
ft_dataset = dc.data.NumpyDataset(X, y)

X = np.random.rand(n_samples, input_size)
y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
test_dataset = dc.data.NumpyDataset(X)


toy = ToyTorchModel(input_size, d_hidden, n_tasks, model_dir = './testfolder')
# toy2 = ToyTorchModel(input_size, hidden_layers, n_tasks, model_dir = './testfolder')

pretrainer = ToyPretrainer(toy) # model_dir = './testfolder/pretrain'
pretrainer.fit(pt_dataset, nb_epoch=100, checkpoint_interval=10) 
# MUST run load_from_pretrained() after fit, load from pretrained loads only embedding. two totally different models

### build new model, fit normally, then load from pretrained with only embedding
# toy.load_from_pretrained(pretrainer, include_top=True, model_dir = './testfolder/pretrain') # works if you run fit, otherwise not bc optimizer params different 
# toy.model.load_state_dict(torch.load(os.path.join(pretrainer.pretrain_model_dir, "checkpoint.pt")))  # works, checkpoint is output from pretrainer.fit

### load_from_pretrained() test, works
# preds = toy2.predict(test_dataset)
# print('toy2 preds: \n', preds)
# # toy.load_from_pretrained(toy2, include_top=False, model_dir = toy2.model_dir)
# preds = toy.predict(test_dataset)
# print('toy preds: \n', preds)