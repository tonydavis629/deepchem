import deepchem as dc
from deepchem.models import TorchModel
from deepchem.models.losses import L2Loss, L1Loss
import torch.nn as nn
import numpy as np


class PretrainableTorchModel(TorchModel):
    @property
    def embedding():
        return NotImplementedError("Subclass must define the embedding")

    def build_embedding(self):
        return NotImplementedError("Subclass must define the embedding")

    def build_head(self):
        return NotImplementedError("Subclass must define the head")

    def build_model(self):
        return NotImplementedError("Subclass must define the model")


class ToyTorchModel(PretrainableTorchModel):
    """Example TorchModel for testing pretraining."""

    def __init__(self, input_dim, d_hidden, d_output, **kwargs):
        self.input_dim = input_dim
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.loss = L2Loss()
        self._head = self.build_head()
        self._embedding = self.build_embedding()
        self._model = self.build_model(self._embedding, self._head)
        super().__init__(self._model, self.loss, **kwargs)

    @property
    def embedding(self):
        return self._embedding

    def build_embedding(self):
        return nn.Linear(self.input_dim, self.d_hidden)

    def build_head(self):
        return nn.Linear(self.d_hidden, self.d_output)

    def build_model(self, embedding, head):
        return nn.Sequential(embedding, head)


class Pretrainer(TorchModel):
    """Abstract pretrainer class. This class is meant to be subclassed for pretraining TorchModels."""

    def __init__(self, torchmodel: PretrainableTorchModel, **kwargs):
        super().__init__(torchmodel.model, torchmodel.loss, **kwargs)

    @property
    def embedding(self):
        return NotImplementedError("Subclass must define the embedding")

    def build_pretrain_loss(self):
        return NotImplementedError("Subclass must define the pretrain loss")


class ToyPretrainer(Pretrainer):
    """Example Pretrainer for testing."""

    def __init__(self, model: ToyTorchModel, pt_tasks: int, **kwargs):

        self._embedding = model.build_embedding()
        self._head = self.build_head(model.d_hidden, pt_tasks)
        self._model = model.build_model(self._embedding, self._head)
        self.loss = self.build_pretrain_loss()
        torchmodel = TorchModel(self._model, self.loss, **kwargs)
        super().__init__(torchmodel, **kwargs)

    @property
    def embedding(self):  # use in load_from_pretrained
        return self._embedding
    
    def build_pretrain_loss(self):
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

toy = ToyTorchModel(input_size, d_hidden, n_tasks, model_dir='./folder1')
toy2 = ToyTorchModel(input_size, d_hidden, n_tasks)

pretrainer = ToyPretrainer(toy, pt_tasks=5, model_dir='./folder2')
pretrainer.fit(pt_dataset, nb_epoch=100, checkpoint_interval=10)

toy2.load_from_pretrained(pretrainer, include_top=False, model_dir='./folder2') #memory issues potentially

# Freeze embedding for finetuning
toy2.embedding.requires_grad = False

toy2.fit(ft_dataset, nb_epoch=100, checkpoint_interval=10)

preds = toy2.predict(test_dataset)
print('toy2 preds: \n', preds)
