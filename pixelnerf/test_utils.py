import torch
from PIL import Image

from pixelnerf.Dataset import sample_rays_np
from pixelnerf.Render import render_rays
import numpy as np
from tqdm import tqdm
import imageio


def rot_phi(phi):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def rot_theta(th):
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def generate_video_nearby(net, ref_dataset, bound, N_samples, device, v_path, r=5.0):
    fx = ref_dataset.fx
    fy = ref_dataset.fy
    img_size = ref_dataset.img_size
    c2w = ref_dataset.c2w[0]
    frames = list()
    for th in tqdm(np.linspace(-1.0, 1.0, 120, endpoint=False)):
        theta = rot_theta(r * np.sin(np.pi * 2.0 * th) / 180.0 * np.pi)
        phi = rot_phi(r * np.cos(np.pi * 2.0 * th) / 180.0 * np.pi)
        rgb = generate_frame(net, theta @ phi @ c2w, fx, fy, img_size, bound, N_samples, device, ref_dataset)
        # frames.append((255 * np.clip(rgb.cpu().numpy(), 0, 1)).astype(np.uint8))
        if rgb.ndim == 3 and rgb.shape[-1] == 3:  # 确保是一个 RGB 图像
            frame1 = (255 * np.clip(rgb.cpu().numpy(), 0, 1)).astype(np.uint8)

            imagee1 = Image.fromarray(frame1)
            imagee1.save('output/1.png')
            frames.append(frame1)
        else:
            print("生成的 rgb 数据不符合预期，检查 generate_frame 函数的输出。")

    imageio.mimwrite(v_path, frames, fps=30, quality=7)


@torch.no_grad()
def generate_frame(net, c2w, fx, fy, img_size, bound, N_samples, device, ref_dataset):
    rays_o, rays_d = sample_rays_np(img_size, img_size, fx, fy, c2w)
    rays_o = torch.tensor(rays_o, device=device)
    rays_d = torch.tensor(rays_d, device=device)
    img_lines = list()
    for i in range(rays_d.shape[0]):
        rays_od = (rays_o[i], rays_d[i])
        img_lines.append(render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, ref=ref_dataset))

    return torch.cat([img[0].unsqueeze(dim=0) for img in img_lines], dim=0)
