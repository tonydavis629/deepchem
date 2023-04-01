"""
Gathers all models in one place for convenient imports
"""
# flake8: noqa
import logging

from deepchem.models.models import Model
from deepchem.models.multitask import SingletaskToMultitask
from deepchem.models.wandblogger import WandbLogger
from deepchem.models.tensorflow_models.callbacks import ValidationCallback

logger = logging.getLogger(__name__)

# Tensorflow Dependency Models
try:
    from deepchem.models.tensorflow_models.keras_model import KerasModel

    from deepchem.models.tensorflow_models.IRV import MultitaskIRVClassifier
    from deepchem.models.tensorflow_models.robust_multitask import RobustMultitaskClassifier
    from deepchem.models.tensorflow_models.robust_multitask import RobustMultitaskRegressor
    from deepchem.models.tensorflow_models.progressive_multitask import ProgressiveMultitaskRegressor, ProgressiveMultitaskClassifier
    from deepchem.models.tensorflow_models.graph_models import WeaveModel, DTNNModel, DAGModel, GraphConvModel, MPNNModel
    from deepchem.models.tensorflow_models.scscore import ScScoreModel

    from deepchem.models.tensorflow_models.seqtoseq import SeqToSeq
    from deepchem.models.tensorflow_models.gan import GAN, WGAN
    from deepchem.models.tensorflow_models.molgan import BasicMolGANModel
    from deepchem.models.tensorflow_models.text_cnn import TextCNNModel
    from deepchem.models.tensorflow_models.atomic_conv import AtomicConvModel
    from deepchem.models.tensorflow_models.chemnet_models import Smiles2Vec, ChemCeption
except ModuleNotFoundError as e:
    logger.warning(
        f'Skipped loading some Tensorflow models, missing a dependency. {e}')

########################################################################################
# Compatibility imports for renamed TensorGraph models. Remove below with DeepChem 3.0.
########################################################################################
try:
    from deepchem.models.tensorflow_models.text_cnn import TextCNNTensorGraph
    from deepchem.models.tensorflow_models.graph_models import WeaveTensorGraph, DTNNTensorGraph, DAGTensorGraph, GraphConvTensorGraph, MPNNTensorGraph
    from deepchem.models.tensorflow_models.IRV import TensorflowMultitaskIRVClassifier
except ModuleNotFoundError:
    pass
