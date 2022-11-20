import torch
import time
import os
import numpy as np
from opt import get_opts
from models.networks import NGP
from models.rendering import render
from metrics import psnr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from train import depth2img
import imageio

if __name__ == "__main__":
    hparams = get_opts()
    kwargs = {'root_dir': hparams.root_dir,
          'split':'test',
          'downsample':1.0/4}
    dataset = dataset_dict[hparams.dataset_name](**kwargs)
    dataset_name = hparams.dataset_name
    scene = hparams.exp_name
    
    model = NGP(scale = hparams.scale).cuda()
    load_ckpt(model, hparams.ckpt_path)

    psnrs = []; ts = []; imgs = []; depths = []
    os.makedirs(f'{scene}_traj', exist_ok=True)

    for img_idx in tqdm(range(len(dataset))):
        t = time.time()
        rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
        results = render(model, rays_o, rays_d,
                     **{'test_time': True,
                        'T_threshold': 1e-2,
                        'exp_step_factor': 1/256})
        torch.cuda.synchronize()
        ts += [time.time()-t]

        pred = results['rgb'].reshape(dataset.img_wh[1], dataset.img_wh[0], 3).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        depth = results['depth'].reshape(dataset.img_wh[1], dataset.img_wh[0]).cpu().numpy()
        depth_ = depth2img(depth)
        imgs += [pred]
        depths += [depth_]
        imageio.imwrite(f'{scene}_traj/{img_idx:03d}.png', pred)
        imageio.imwrite(f'{scene}_traj/{img_idx:03d}_d.png', depth_)

        if dataset.split != 'test_traj':
            rgb_gt = dataset[img_idx]['rgb'].cuda()
            psnrs += [psnr(results['rgb'], rgb_gt).item()]
    if psnrs: print(f'mean PSNR: {np.mean(psnrs):.2f}, min: {np.min(psnrs):.2f}, max: {np.max(psnrs):.2f}')
    print(f'mean time: {np.mean(ts):.4f} s, FPS: {1/np.mean(ts):.2f}')
    print(f'mean samples per ray: {results["total_samples"]/len(rays_d):.2f}')

    if True:
        imageio.mimsave(f'{scene}_traj/rgb.mp4', imgs, fps=5)
        imageio.mimsave(f'{scene}_traj/depth.mp4', depths, fps=5)