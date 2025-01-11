import glob
import os
import re

import torch
from tqdm import tqdm
from Dataset import get_data
from torch.utils.data import DataLoader
from Network import PixelNeRF
from Render import render_rays
from test_utils import generate_video_nearby
import numpy as np

def restore_checkpoint(
        checkpoint_dir,
        net,
        optimizer
):
    dirs = glob.glob(os.path.join(checkpoint_dir, "*"))
    dirs.sort()
    path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        init_step = 0
    else:
        checkpoint = torch.load(path)  # 这里需要使用 torch.load
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 提取文件名中的最后一个数字
        match = re.search(r'(\d+)', os.path.basename(path))
        if match:
            init_step = int(match.group(1))  # 提取并转换为整数
        else:
            raise ValueError("Invalid checkpoint filename format: no number found")

    if init_step == 0:
        print("Checkpoint does not exist. Starting a new training run.")
    else:
        print("Resuming from checkpoint {}".format(init_step))
    return init_step



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(999)
np.random.seed(666)
# n_train = 3

#############################
# create rays for batch train
#############################
print("Process rays data for training!")
rays_dataset, ref_dataset, bounds = get_data("data/360_v2/bonsai", device, factor=4, mode='train')

#############################
# training parameters
#############################
Batch_size = 2048
rays_loader = DataLoader(rays_dataset, batch_size=Batch_size, drop_last=True, shuffle=True)
print(f"Batch size of rays: {Batch_size}")

# bounds = (2., 6.)
N_samples = (64, None)
epoch = 1300
img_f_ch = 512
lr = 1e-4
#############################
# training
#############################
net = PixelNeRF(img_f_ch).to(device)
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
save_dir = 'checkpoints/bonsai'
save_interval = 100
init_step = restore_checkpoint("checkpoints/bonsai", net, optimizer)
print("Start Training!")
dataiter = iter(rays_loader)
for e in range(init_step, epoch):
    with tqdm(total=len(rays_loader), desc=f"Epoch {e+1}", ncols=100) as p_bar:
        for train_rays in rays_loader:
            assert train_rays.shape == (Batch_size, 9)
            rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
            rays_od = (rays_o, rays_d)
            rgb, _, __ = render_rays(net, rays_od, bound=bounds, N_samples=N_samples, device=device, ref=ref_dataset)
            loss = mse(rgb, target_rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
            p_bar.update(1)

    if (e + 1) % save_interval == 0 and e != 0:
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{e+1}.pt')
        torch.save({
            'epoch': e+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

print('Finish Training!')

# print('Start Generating Video!')
# net.eval()
# generate_video_nearby(net, ref_dataset, bounds, N_samples, device, './video/test.mp4')
# print('Finish Generating Video!')

