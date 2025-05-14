import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
# os.chdir('/root/myWorkPlace/MTSDM/MTSDM/Util/CleanDiffuser-main')
sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/Util/CleanDiffuser-main")
sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/model/source-Diff-MTS")
import data.CMAPSSDataset as CMAPSSDataset
from DiffusionFreeGuidence.Unet1D_TD import UNet1D_TD

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion import DiscreteDiffusionSDE, ContinuousDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS

# Constants
DEVICE = 'cpu'
EPSILON = 1e-3

# Fixtures for Network Architecture Initialization
# @pytest.fixture
def init_network():
    obs_dim = 10  # Example dimension
    args = {
        'emb_dim': 64,
        'd_model': 128,
        'n_heads': 8,
        'depth': 2,
        'label_dropout': 0.1,
        'task': {'horizon': 5}
    }

    nn_diffusion = DiT1d(
        obs_dim, emb_dim=args['emb_dim'],
        d_model=args['d_model'], n_heads=args['n_heads'], depth=args['depth'], timestep_emb_type="fourier"
    )
    nn_condition = MLPCondition(
        in_dim=1, out_dim=args['emb_dim'], hidden_dims=[args['emb_dim']], act=nn.SiLU(), dropout=args['label_dropout']
    )

    fix_mask = torch.zeros((args['task']['horizon'], obs_dim))
    fix_mask[0] = 1.
    loss_weight = torch.ones((args['task']['horizon'], obs_dim))
    loss_weight[1] = 10

    return nn_diffusion, nn_condition, fix_mask, loss_weight

# Fixtures for Diffusion Models
def test_diffusion_sde(init_network):
    nn_diffusion, nn_condition, fix_mask, loss_weight = init_network

    model = DiscreteDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        device=DEVICE
    )

    model2 = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        device=DEVICE
    )

if __name__ == "__main__":

    datasets = CMAPSSDataset.CMAPSSDataset(fd_number='FD001', sequence_length=48 ,deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_sensor_slice(train_data), datasets.get_label_slice(train_data)    
    print(f"train_data.shape: {train_data.shape}")
    print(f"train_label.shape: {train_label.shape}")
    '''
    train_data.shape: torch.Size([15931, 48, 14])
    train_label.shape: torch.Size([15931, 1])
    '''

    model = DiscreteDiffusionSDE(
        nn_diffusion=UNet1D_TD(dim = 32, dim_mults = (1, 2, 2), cond_drop_prob = 0.2, channels = 14),
        nn_condition=None,
        fix_mask=None,
        loss_weight=None,
        device=DEVICE
    )

    samples, _ = model.sample(
        prior=torch.zeros(train_data.shape[0], 14, 48),
        solver = 'ode_dpmsolver_1',
        nsamples=train_data.shape[0],
        sample_steps=5,
        use_ema = False,
        condition_cfg = train_label,
    )

    print(f"sample.shape: {samples.shape}")


    
    
