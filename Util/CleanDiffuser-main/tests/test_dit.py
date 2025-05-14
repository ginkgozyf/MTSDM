import sys

sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/Util/CleanDiffuser-main")
sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/model/source-Diff-MTS")
import data.CMAPSSDataset as CMAPSSDataset
import pytest
import torch
from cleandiffuser.nn_diffusion.dit import DiT1d, DiT1Ref, DiTBlock, FinalLayer1d

def test_dit_block_forward():
    block = DiTBlock(hidden_size=128, n_heads=4, dropout=0.1)
    x = torch.randn(2, 10, 128)
    t = torch.randn(2, 128)
    result = block(x, t)
    assert result.shape == x.shape

def test_final_layer1d_forward():
    layer = FinalLayer1d(hidden_size=128, out_dim=64)
    x = torch.randn(2, 10, 128)
    t = torch.randn(2, 128)
    result = layer(x, t)
    assert result.shape == (2, 10, 64)

def test_dit1d_forward():
    model = DiT1d(in_dim=64, emb_dim=128, d_model=128, n_heads=4, depth=2, dropout=0.1)
    x = torch.randn(2, 10, 64)
    noise = torch.randn(2)
    condition = torch.randn(2, 128)
    result = model(x, noise, condition)
    assert result.shape == x.shape

def test_dit1ref_forward():
    model = DiT1Ref(in_dim=64, emb_dim=128, d_model=128, n_heads=4, depth=2, dropout=0.1)
    x = torch.randn(2, 10, 128)  # in_dim * 2
    noise = torch.randn(2)
    condition = torch.randn(2, 128)
    result = model(x, noise, condition)
    assert result.shape == (2, 10, 128)

if __name__ == "__main__":
    datasets = CMAPSSDataset.CMAPSSDataset(fd_number='FD001', sequence_length=48 ,deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_sensor_slice(train_data), datasets.get_label_slice(train_data)    
    print(f"train_data.shape: {train_data.shape}")
    print(f"train_label.shape: {train_label.shape}")

    model = DiT1d(in_dim=14, emb_dim=128, d_model=128, n_heads=4, depth=2, dropout=0.1)
    x = train_data
    t = torch.randn(train_data.shape[0])
    condition = train_label
    result = model(x, t, condition.expand(-1, 128))
    print(f"result.shape: {result.shape}")
    # test_dit1d_forward()