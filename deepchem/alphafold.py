
sequence = 'MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH'
num_residues = len(sequence)

# %% [markdown]
# #### Search against genetic databases
# 
# Use jackhmmr to identify homologous protein structures 

# %%
from deepchem.utils.alphafold_utils.jackhammer import Jackhmmer

msas, deletion_matrices = Jackhmmer(sequence, dbs = ['uniref90', 'smallbfd', 'mgnify'])

# #### Process the features
from deepchem.feat.sequence_featurizers import template_featurizer, data_pipeline, feature_pipeline

template_feat = template_featurizer(mmcif_dir = '/data/mmcif', obsolete_pdbs_path = '/data/obs_pdbs', max_template_date = '2100-01-01', max_hits = 20) # optional

data_pipe = data_pipeline(template_featurizer = template_feat) #templates are optional

feature_dict = {}
feature_dict.update(data_pipe.make_sequence_features(sequence, 'test', num_residues))
feature_dict.update(data_pipe.make_msa_features(msas, deletion_matrices=deletion_matrices))

feat_pipe = feature_pipeline(config='default')
processed_feature_dict = feat_pipe.process_features(feature_dict)

# %% [markdown]
# #### Create the model

# %%
# from deepchem.models.layers import EvoformerLayer
# import torch
# from torch import nn
# from pytorch_lightning import LightningModule, Trainer, LightningDataModule
# ALPHAFOLD_PARAM_SOURCE_URL = 'https://storage.googleapis.com/alphafold/alphafold_params_2022-01-19.tar'

class EvoFormerStack(nn.Module):
  """Stack of EvoFormer layers."""

  def __init__(self, num_layers, num_heads, d_model, d_ff, dropout_rate):
    super().__init__()
    self.layers = nn.ModuleList([
        EvoformerLayer(num_heads, d_model, d_ff, dropout_rate)
        for _ in range(num_layers)
    ])

  def forward(self, x, msa, mask):
    for layer in self.layers:
      x = layer(x, msa, mask)
    return x

class AlphaFold(nn.Module): # or nn.module and create lightning wrapper?
  def __init__(self,config):
    """
    Parameters:
    
    config: dict
      Model configuration
    """
  def download_alphafold_params(self, url=ALPHAFOLD_PARAM_SOURCE_URL):
    """Downloads AlphaFold parameters from a URL."""
    
  def iteration(self, feats, prevs, recycle=True):
    """Runs a single iteration of the model."""
    pass
  def forward(self,batch):
    """
    Parameters:
    
    batch: dict
      Batch of data
    """
    extract data from batch
    pass through model
    recycle data
    return output
    pass
    


# %% [markdown]
# #### Run the model

# %%
chemfold = AlphaFold(config='default')
chemfold.to('cuda')

with torch.no_grad():
  chemfold.eval()
  output = chemfold(processed_feature_dict)
