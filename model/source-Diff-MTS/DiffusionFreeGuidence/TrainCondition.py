
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset,random_split
import numpy as np

from DiffusionFreeGuidence import GaussianDiffusionSampler, GaussianDiffusionTrainer 
from DiffusionFreeGuidence.Unet1D_TD import UNet1D_TD
from DiffusionFreeGuidence.Diffwave import DiffWave
from DiffusionFreeGuidence.tabddpm import MLPDiffusion
import sys
sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/Util/CleanDiffuser-main")
from cleandiffuser.nn_diffusion.dit import DiT1d, DiT1Ref, DiTBlock, FinalLayer1d

from Scheduler import GradualWarmupScheduler
from DiffusionFreeGuidence.diffwaveimputer import DiffWaveImputer
from DiffusionFreeGuidence.SSSD import SSSDS4Imputer
from DiffusionFreeGuidence.csdi import diff_CSDI
import data.CMAPSSDataset as CMAPSSDataset
import wandb

def train(args, train_data, train_label):
    device = args.device
    best_loss = 9999

    train_dataset = TensorDataset(train_data.permute(0,2,1).to(device), train_label.to(device))
    dataloader= DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    
    # model setup
    if args.model_name == 'DiffUnet': 
        net_model = UNet1D_TD(dim = 32, dim_mults = (1, 2, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
    if args.model_name == 'DiffWave': 
        net_model = DiffWaveImputer(seq_length=args.window_size)
    if args.model_name == 'SSSD': 
        net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
    if args.model_name == 'tabddpm':     
        net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.3}, dim_t = 512)
    if args.model_name == 'csdi': 
        net_model = diff_CSDI(inputdim=14)
    if args.model_name == 'Dit':
        net_model = DiT1d(in_dim=14, emb_dim=128, d_model=128, n_heads=4, depth=2, dropout=0.1)
                 
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier,
                                             warm_epoch=args.epoch // 10 + 1, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, beta_1=1e-4 ,beta_T=0.028, T=args.T, schedule_name = args.schedule_name, loss_type=args.loss_type ).to(device)

    # start training
    for e in range(args.epoch):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            loss_list=[]
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), args.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": sum(loss_list)/len(loss_list),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        current_loss = sum(loss_list)/len(loss_list)
        wandb.log({"Diffusion_Loss":current_loss})
        if e > 5 and current_loss < best_loss:
            torch.save(net_model.state_dict(), args.model_path)
            print('*******imporove!!!********')

def sample(args, train_label = None):
    if train_label == None:
        datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size ,deleted_engine=[1000])
        train_data = datasets.get_train_data()
        train_label = datasets.get_label_slice(train_data)        
    device = args.device
    # load model and evaluate
    with torch.no_grad():
        if args.model_name == 'DiffUnet': 
            net_model = UNet1D_TD(dim = 32, dim_mults = (1, 2, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
        if args.model_name == 'DiffWave': 
            net_model = DiffWaveImputer(seq_length=args.window_size)
        if args.model_name == 'SSSD': 
            net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
        if args.model_name == 'tabddpm':     
            net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.0}, dim_t = 512)
        if args.model_name == 'csdi':     
            net_model = diff_CSDI(inputdim=14)
        if args.model_name == 'Dit':
            net_model = DiT1d(in_dim=14, emb_dim=128, d_model=128, n_heads=4, depth=2, dropout=0.1)
                     
        ckpt = torch.load(args.model_path)
        net_model.load_state_dict(ckpt)
        print("model load weight done.")
        net_model.eval()
        sampler = GaussianDiffusionSampler(
            net_model,  beta_1=1e-4 ,beta_T=0.028, T=args.T, w=args.w, schedule_name = args.schedule_name).to(device)
        # Sampled from standard normal distribution
        noisydata = torch.randn(
            size=[train_label.shape[0], args.input_size, args.window_size], device=device)
        if args.sample_type == 'ddpm':
            sampledata = sampler(noisydata , train_label.to(device)).permute(0,2,1)
            print("sampledata shape: ",sampledata.shape)      
        if args.sample_type == 'ddim':
            sampledata = sampler.sample_backward(noisydata ,device,  train_label.to(device)).permute(0,2,1)  
            print("sampledata shape: ",sampledata.shape)
        if args.sample_type == 'dpmsolver':
            sampledata = sampler.sample_data(x = noisydata, classes = train_label.to(device), sample_type = 'dpmsolver').permute(0,2,1)
            print("sampledata shape: ",sampledata.shape)
        if args.sample_type == 'dpmsolver++':
            sampledata = sampler.sample_data(x = noisydata, classes = train_label.to(device), sample_type = 'dpmsolver++').permute(0,2,1)
            print("sampledata shape: ",sampledata.shape)
        if args.sample_type == 'unipc':
            sampledata = sampler.sample_data(x = noisydata, classes = train_label.to(device), sample_type = 'unipc').permute(0,2,1)
            print("sampledata shape: ",sampledata.shape)
        np.savez(args.syndata_path,data=sampledata.cpu().numpy(), label =  train_label.cpu().numpy())
    return sampledata