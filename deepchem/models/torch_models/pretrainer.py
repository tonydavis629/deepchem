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
                 **kwargs): 
        # self.pretrain_model_dir = os.path.join(torchmodel.model_dir, "pretrain")
        super().__init__(torchmodel.model, torchmodel.loss,**kwargs) 
    def freeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = False
    def unfreeze_embedding(self):
        self.torchmodel.model.embedding.weight.requires_grad = True
    def _define_pretrain_loss(self):
        return NotImplementedError("Subclass must define the pretrain loss")

class ToyPretrainer(Pretrainer): # diamond inheritance    
    def __init__(self, 
                 model:ToyTorchModel,
                 pt_tasks:int,
                 **kwargs): 
        self.head = self.build_head(model.embedding.out_features, pt_tasks)
        self.embedding = model.embedding
        self.model = model.build_model(model.embedding,model.head)
        model.loss = self._define_pretrain_loss()
        super().__init__(model, model_dir = model.model_dir, **kwargs)
    def _define_pretrain_loss(self):
        return L1Loss()
    def build_head(self, d_hidden, pt_tasks): #can overwrite build_head
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


toy = ToyTorchModel(input_size, d_hidden, n_tasks, model_dir = './testfolder')
toy2 = ToyTorchModel(input_size, d_hidden, n_tasks, model_dir = './testfolder')

pretrainer = ToyPretrainer(toy, pt_tasks=5) # model_dir = './testfolder/pretrain'
pretrainer.fit(pt_dataset, nb_epoch=100, checkpoint_interval=10) 
# MUST run load_from_pretrained() after fit, load from pretrained loads only embedding. two totally different models

### build new model, fit normally, then load from pretrained with only embedding
toy2.load_from_pretrained(pretrainer, include_top=False, model_dir = './testfolder') # works if you run fit, otherwise not bc optimizer params different 
# toy.model.load_state_dict(torch.load(os.path.join(pretrainer.pretrain_model_dir, "checkpoint.pt")))  # works, checkpoint is output from pretrainer.fit

### load_from_pretrained() test, works
# preds = toy2.predict(test_dataset)
# print('toy2 preds: \n', preds)
# # toy.load_from_pretrained(toy2, include_top=False, model_dir = toy2.model_dir)
# preds = toy.predict(test_dataset)
# print('toy preds: \n', preds)