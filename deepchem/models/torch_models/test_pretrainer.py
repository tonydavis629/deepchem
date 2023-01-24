import os
import pytest
import deepchem as dc
import numpy as np
import math
import unittest
import torch
from pretrainer import Pretrainer, PretrainableTorchModel
import torch.nn as nn
from torch_model import TorchModel


@pytest.mark.torch
def test_overfit_subclass_model():
    """Test fitting a TorchModel defined by subclassing Module."""
    n_data_points = 10
    n_features = 2
    np.random.seed(1234)
    X = np.random.rand(n_data_points, n_features)
    y = (X[:, 0] > X[:, 1]).astype(np.float32)
    dataset = dc.data.NumpyDataset(X, y)
    
    class ExampleTorchModel(PretrainableTorchModel):
        """Example TorchModel for testing pretraining."""

        def __init__(self, input_dim, d_hidden, d_output, **kwargs):
            self.input_dim = input_dim
            self.d_hidden = d_hidden
            self.d_output = d_output
            self.loss = dc.models.losses.L2Loss()
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
        
    class ExamplePretrainer(Pretrainer):
        """Example Pretrainer for testing."""

        def __init__(self, model: ExampleTorchModel, pt_tasks: int, **kwargs):

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
            return dc.models.losses.SigmoidCrossEntropy()

        def build_head(self, d_hidden, pt_tasks):
            return nn.Linear(d_hidden, pt_tasks)

    input_dim = 10
    d_hidden = 3
    d_output = 4
    pt_tasks = 7
    example_model = ExampleTorchModel(input_dim, d_hidden, d_output)
    pretrainer = ExamplePretrainer(example_model, pt_tasks)

    pretrainer.fit(dataset, nb_epoch=1000)
    prediction = np.squeeze(pretrainer.predict_on_batch(X))
    assert np.array_equal(y, np.round(prediction))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    scores = pretrainer.evaluate(dataset, [metric])
    assert scores[metric.name] > 0.9
