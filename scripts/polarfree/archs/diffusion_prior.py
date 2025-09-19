import torch
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from functools import partial
import torch.nn as nn

from polarfree.utils.base_model import BaseModel
from polarfree.utils.beta_schedule import make_beta_schedule, default
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from copy import deepcopy
class diffusion_prior(nn.Module):
    """PolarFree Stage 2 model for polarization image enhancement."""

    def __init__(self, opt):
        super(diffusion_prior, self).__init__()
        # Setup diffusion model
        self.apply_ldm = False
        self.opt = opt
        #self.schedule_opt = self.opt['diffusion_schedule']
        self.set_new_noise_schedule(self.opt['diffusion_schedule'],  torch.device('cuda'))


    def set_new_noise_schedule(self, schedule_opt, device):
        """Set up noise schedule for diffusion model."""
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # Create beta schedule
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['timesteps'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # Register buffers for diffusion process
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Posterior calculations
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        )
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        )

    def predict_start_from_noise(self, x_t, t, noise):
            """Predict x0 from noise."""
            return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
            """Compute posterior q(x_{t-1} | x_t, x_0)."""
            posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
            posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
            return posterior_mean, posterior_log_variance_clipped

    def print_different_keys_loading(self,crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self,net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self.print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def load_network_with_path(self,network, load_path, network_label):
        """Helper to load a specific network."""
        param_key = self.opt['path'].get(f'param_key_{network_label}', 'params')
        strict_load = self.opt['path'].get(f'strict_load_{network_label}', True)
        self.load_network(network, load_path, strict_load, param_key)

    def p_mean_variance(self, x, t, net_d,clip_denoised=True, condition_x=None, ema_model=False):
        """Compute mean and variance of p(x_{t-1} | x_t)."""
        if condition_x is None:
            raise RuntimeError('Must have LQ/LR condition')

        t_tensor = torch.full(x.shape, t.item() + 1, device=x.device, dtype=torch.long)
        '''net_d = build_network(self.opt['network_d'])
        net_d.to(torch.device('cuda'))
        load_path = self.opt['path'].get('pretrain_network_d', None)
        self.load_network_with_path(self.opt,net_d, load_path, 'd')'''
        net_d.eval()
        noise = net_d(x, condition_x, t_tensor)
        x_recon = self.predict_start_from_noise(x, t=t.item(), noise=noise)

        if clip_denoised:
            x_recon.clamp(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t.item())
        return ({
            "mean": model_mean,
            "log_variance": posterior_log_variance,
            "pred_xstart": x_recon,
        })

    def forward(self, x, t,net_d,condition_x):
        out = self.p_mean_variance(x,t,net_d,condition_x=condition_x)
        return out

class diffusion_prior_2(nn.Module):
    """PolarFree Stage 2 model for polarization image enhancement."""

    def __init__(self, opt):
        super(diffusion_prior_2, self).__init__()
        # Setup diffusion model
        self.apply_ldm = False
        self.opt = opt
        #self.schedule_opt = self.opt['diffusion_schedule']
        self.set_new_noise_schedule(self.opt['diffusion_schedule'],  torch.device('cuda'))
        self.num_timesteps = self.opt['diffusion_schedule']['timesteps']

    def set_new_noise_schedule(self, schedule_opt, device):
        """Set up noise schedule for diffusion model."""
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # Create beta schedule
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['timesteps'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))


        # Register buffers for diffusion process
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Posterior calculations
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        )
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        )

    def predict_start_from_noise(self, x_t, t, noise):
            """Predict x0 from noise."""
            return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
            """Compute posterior q(x_{t-1} | x_t, x_0)."""
            posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
            posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
            return posterior_mean, posterior_log_variance_clipped

    def print_different_keys_loading(self,crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self,net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self.print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def load_network_with_path(self,network, load_path, network_label):
        """Helper to load a specific network."""
        param_key = self.opt['path'].get(f'param_key_{network_label}', 'params')
        strict_load = self.opt['path'].get(f'strict_load_{network_label}', True)
        self.load_network(network, load_path, strict_load, param_key)

    def p_mean_variance(self, x, t, net_d,clip_denoised=True, condition_x=None, ema_model=False):
        """Compute mean and variance of p(x_{t-1} | x_t)."""
        if condition_x is None:
            raise RuntimeError('Must have LQ/LR condition')

        t_tensor = torch.full(x.shape, t + 1, device=x.device, dtype=torch.long)
        '''net_d = build_network(self.opt['network_d'])
        net_d.to(torch.device('cuda'))
        load_path = self.opt['path'].get('pretrain_network_d', None)
        self.load_network_with_path(self.opt,net_d, load_path, 'd')'''
        net_d.eval()
        noise = net_d(x, condition_x, t_tensor)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance,x_recon

    def p_sample(self, x, t, net_d,clip_denoised=True, condition_x=None, ema_model=False):
        """Sample from p(x_{t-1} | x_t)."""
        model_mean, posterior_log_variance,x_recon = self.p_mean_variance(
            x=x, t=t, net_d=net_d, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return ({
            "mean": model_mean,
            "log_variance": posterior_log_variance,
            "pred_xstart": x_recon,
        })

    def p_sample_loop(self, x_in, x_noisy,net_d, ema_model=False):
        """Run full reverse process."""
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            out = self.p_sample(img, i, net_d, condition_x=x_in, ema_model=ema_model)
            yield out
            img = out["mean"]


    def forward(self, condition_x, x,net_d):
        for sample in self.p_sample_loop(condition_x, x,net_d):
            out = sample["mean"]
        return out

