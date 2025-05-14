import sys

sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/Util/CleanDiffuser-main")
sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/model/source-Diff-MTS")
sys.path.append("/root/myWorkPlace/MTSDM/MTSDM/sample/DPM-Solver-v3-main/codebases/guided-diffusion")
import data.CMAPSSDataset as CMAPSSDataset
from cleandiffuser.nn_diffusion.dit import DiT1d, DiT1Ref, DiTBlock, FinalLayer1d

import torch
import torch.nn.functional as F
import math
import numpy as np
import tqdm


class Sampler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_beta_schedule(self, beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "cosine":
            return betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas


    def sample_data(self, x, classes = None, model = None, sample_type = "unipc"):
        device = self.device
        self.x = x
        self.classes = classes
        self.model = model
        self.sample_type = sample_type
        self.thresholding = None
        self.denoise = None
        self.timesteps = 20
        self.order = 2
        self.skip_type = "time_uniform"
        self.method = "multistep"
        self.lower_order_final = True

        betas = self.get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        if True:
            model_kwargs = {}
        else:
            model_kwargs = {"condition": classes}

        if self.sample_type in ["dpmsolver", "dpmsolver++"]:
            from samplers.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

            def model_fn(x, t, cond, **model_kwargs):
                out = model(x, t, cond, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if False:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn_continuous = model_wrapper(
                model,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="classifier-free",
                condition=classes,
                guidance_scale=1.0,
                # classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            dpm_solver = DPM_Solver(
                model_fn_continuous,
                noise_schedule,
                algorithm_type=self.sample_type,
                correcting_x0_fn="dynamic_thresholding" if self.thresholding else None,
            )
            x = dpm_solver.sample(
                x,
                steps=(self.timesteps - 1 if self.denoise else self.timesteps),
                order=self.order,
                skip_type=self.skip_type,
                method=self.method,
                lower_order_final=self.lower_order_final,
                denoise_to_zero=self.denoise,
            )
        elif self.sample_type == "unipc":
            from samplers.uni_pc import NoiseScheduleVP, model_wrapper, UniPC

            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn_continuous = model_wrapper(
                model,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="classifier-free",
                condition=classes,
                guidance_scale=1.0,
                # classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            unipc = UniPC(
                model_fn_continuous,
                noise_schedule,
                algorithm_type="noise_prediction",
                correcting_x0_fn="dynamic_thresholding" if self.thresholding else None,
            )
            x = unipc.sample(
                x,
                # steps=(self.timesteps - 1 if self.denoise else self.timesteps),
                order=self.order,
                skip_type=self.skip_type,
                method=self.method,
                lower_order_final=self.lower_order_final,
                denoise_to_zero=self.denoise,
            )
        else:
            raise NotImplementedError
        return x, classes

if __name__ == "__main__":
    datasets = CMAPSSDataset.CMAPSSDataset(fd_number='FD001', sequence_length=48 ,deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_sensor_slice(train_data), datasets.get_label_slice(train_data)    
    print(f"train_data.shape: {train_data.shape}")
    print(f"train_label.shape: {train_label.shape}")

    model = DiT1d(in_dim=14, emb_dim=128, d_model=128, n_heads=4, depth=2, dropout=0.1)
    # x = train_data
    # t = torch.randn(train_data.shape[0])
    # condition = train_label
    # result = model(x, t, condition.expand(-1, 128))
    # print(f"result.shape: {result.shape}")
    sampler = Sampler()

    noisydata = torch.randn(
            size=[train_label.shape[0], 14, 48])

    x, classes = sampler.sample_data(x = noisydata , classes = train_label, model = model)

    print(f"x.shape: {x.shape}")