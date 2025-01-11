import glob
import logging
import os
import shutil
import sys
import torchvision.transforms as T
import cv2
import numpy as np
import random
import torch.nn.functional as F
import time

from absl import app
import gin
from omegaconf import OmegaConf

from diffusion.ldm.models.diffusion.ddpm import LatentDiffusionSampler
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
from internal import checkpoints
import torch
import accelerate
import tensorboardX
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils._pytree import tree_map
from pixelnerf.Network import PixelNeRF
from diffusion.ldm.util import instantiate_from_config
from internal.datasets import ReferenceDataset
from pixelnerf.ImageEncoder import ImageEncoder, ModifiedPixelNeRF
from pixelnerf.test_utils import generate_frame

configs.define_common_flags()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.
H = 512
img_f_ch = 512
N_samples = (64, None)
bounds =[0.01, 1]
diffusion_config = "diffusion_config/2024-11-29T21-40-17-project.yaml"
diffusion_ckpt = "diffusion_checkpoints/epoch=000018.ckpt"
diff_outdir = "diff_outputs/txt2img-samples"
diff_batchsize = 1


def generate_image_bypose(net, ref_dataset, bound, N_samples, device, pose):
    fx = ref_dataset.fx
    fy = ref_dataset.fy
    img_size = ref_dataset.img_size
    rgb = generate_frame(net, pose, fx, fy, img_size, bound, N_samples, device, ref_dataset).unsqueeze(0).permute(0, 3, 1, 2)
    encoder = ImageEncoder().to(device).eval()
    with torch.no_grad():
        reference_feature = encoder(rgb)

    return reference_feature, rgb


def normalize_camera_poses(poses, near_plane=0.5):
    camera_centers = poses[:, :3, 3]
    focus_point = np.mean(camera_centers, axis=0)

    # 平移：使焦点为原点
    poses[:, :3, 3] -= focus_point

    # 缩放：将相机中心的距离归一化
    max_distance = np.max(np.linalg.norm(poses[:, :3, 3], axis=-1))
    scale = near_plane / max_distance
    poses[:, :3, 3] *= scale

    return poses


# 加载pixelnerf的权重
def get_checkpoint(checkpoint_dir, net, optimizer):
    dirs = glob.glob(os.path.join(checkpoint_dir, "*"))
    dirs.sort()
    path = dirs[-1] if len(dirs) > 0 else None
    checkpoint = torch.load(path)  # 这里需要使用 torch.load
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# 加载diffusion的权重
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join("exp", config.exp_name)
    config.checkpoint_dir = os.path.join(config.exp_path, 'checkpoints')
    utils.makedirs(config.exp_path)
    with utils.open_file(os.path.join(config.exp_path, 'config.gin'), 'w') as f:
        f.write(gin.config_str())

    # accelerator for DDP
    accelerator = accelerate.Accelerator()

    # setup logger
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(config.exp_path, 'log_train.txt'))],
        level=logging.INFO,
    )
    sys.excepthook = utils.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    if config.batch_size % accelerator.num_processes != 0:
        config.batch_size -= config.batch_size % accelerator.num_processes != 0
        logger.info('turn batch size to', config.batch_size)

    # Set random seed.
    accelerate.utils.set_seed(config.seed, device_specific=True)
    # setup model and optimizer
    model = models.Model(config=config)
    optimizer, lr_fn = train_utils.create_optimizer(config, model)

    # load dataset
    dataset = datasets.load_dataset('train', config.data_dir, config)
    print(type(dataset))  # 检查 dataset 的类型
    test_dataset = datasets.load_dataset('test', config.data_dir, config)

    # 为pixelnerf做准备
    net = PixelNeRF(img_f_ch).to(accelerator.device)
    pixel_optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    get_checkpoint("pixelnerf_checkpoints/bonsai", net, pixel_optimizer)
    net.eval()
    # 准备数据
    conference_images = dataset.images
    conference_images = torch.from_numpy(conference_images)
    conference_poses = dataset.poses
    conference_poses = conference_poses.astype(np.float32)
    fx = dataset.fx
    fy = dataset.fy
    conference_images = conference_images.permute(0, 3, 1, 2)  # 改变维度顺序为 (batch_size, channels, height, width)
    # 使用 interpolate 调整大小
    resized_images = F.interpolate(conference_images, size=(512, 512), mode='bilinear', align_corners=False)
    resized_images = resized_images.to(accelerator.device)

    encoder = ImageEncoder().to(accelerator.device).eval()
    with torch.no_grad():
        reference_feature = encoder(resized_images)
    ref_dataset = ReferenceDataset(reference_feature, conference_poses, fx, fy, H)

    # 为diffusion model做准备
    diff_config = OmegaConf.load(f"{diffusion_config}")
    diffusion_model = load_model_from_config(diff_config, f"{diffusion_ckpt}")
    diffusion_model = diffusion_model.to(accelerator.device)
    diffusion_model.eval()
    sampler = LatentDiffusionSampler(diffusion_model)
    os.makedirs(diff_outdir, exist_ok=True)
    diff_outpath = diff_outdir
    sample_path = os.path.join(diff_outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    tmin, tmax = 0, 500

    generator = model.generator
    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             num_workers=8,
                                             shuffle=True,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             persistent_workers=True,
                                             generator=generator,
                                             )
    test_dataloader = torch.utils.data.DataLoader(np.arange(len(test_dataset)),
                                                  num_workers=4,
                                                  shuffle=False,
                                                  batch_size=1,
                                                  persistent_workers=True,
                                                  collate_fn=test_dataset.collate_fn,
                                                  generator=generator,
                                                  )
    if config.rawnerf_mode:
        postprocess_fn = test_dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z, _=None: z

    # use accelerate to prepare.
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

    if config.resume_from_checkpoint:
        init_step = checkpoints.restore_checkpoint(config.checkpoint_dir, accelerator, logger)
    else:
        init_step = 0

    module = accelerator.unwrap_model(model)
    dataiter = iter(dataloader)
    test_dataiter = iter(test_dataloader)

    num_params = train_utils.tree_len(list(model.parameters()))
    logger.info(f'Number of parameters being optimized: {num_params}')

    if (dataset.size > module.num_glo_embeddings and module.num_glo_features > 0):
        raise ValueError(f'Number of glo embeddings {module.num_glo_embeddings} '
                         f'must be at least equal to number of train images '
                         f'{dataset.size}')

    # metric handler
    metric_harness = image.MetricHarness()

    # tensorboard
    if accelerator.is_main_process:
        summary_writer = tensorboardX.SummaryWriter(config.exp_path)
        # function to convert image for tensorboard
        tb_process_fn = lambda x: x.transpose(2, 0, 1) if len(x.shape) == 3 else x[None]

        if config.rawnerf_mode:
            for name, data in zip(['train', 'test'], [dataset, test_dataset]):
                # Log shutter speed metadata in TensorBoard for debug purposes.
                for key in ['exposure_idx', 'exposure_values', 'unique_shutters']:
                    summary_writer.add_text(f'{name}_{key}', str(data.metadata[key]), 0)
    logger.info("Begin training...")
    step = init_step + 1
    total_time = 0
    total_steps = 0
    reset_stats = True
    if config.early_exit_steps is not None:
        num_steps = config.early_exit_stepsc
    else:
        num_steps = config.max_steps
    # init_step = 0
    with logging_redirect_tqdm():
        tbar = tqdm(range(init_step + 1, num_steps + 1),
                    desc='Training', initial=init_step, total=num_steps,
                    disable=not accelerator.is_main_process)
        for step in tbar:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch = next(dataiter)
            batch = accelerate.utils.send_to_device(batch, accelerator.device)
            if reset_stats and accelerator.is_main_process:
                stats_buffer = []
                train_start_time = time.time()
                reset_stats = False

            # use lr_fn to control learning rate
            learning_rate = lr_fn(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # fraction of training period
            train_frac = np.clip((step - 1) / (config.max_steps - 1), 0, 1)

            # Indicates whether we need to compute output normal or depth maps in 2D.
            compute_extras = (config.compute_disp_metrics or config.compute_normal_metrics)
            optimizer.zero_grad()
            with accelerator.autocast():
                renderings, ray_history = model(
                    True,
                    batch,
                    train_frac=train_frac,
                    compute_extras=compute_extras,
                    zero_glo=False)

            # 从测试数据中取一个 batch
            try:
                test_batch = next(test_dataiter)
            except StopIteration:
                test_dataiter = iter(test_dataloader)  # 重新初始化测试数据迭代器
                test_batch = next(test_dataiter)
            target_pose = test_batch['pose']  # 选择当前目标位姿
            target_pose = target_pose.numpy()
            target_pose = np.vstack((target_pose, np.array([0, 0, 0, 1])))
            target_pose = target_pose.astype(np.float32)
            reference_feature, target_rgb = generate_image_bypose(net, ref_dataset, bounds, N_samples, accelerator.device, target_pose)
            encoder = ModifiedPixelNeRF().to(accelerator.device).eval()
            with torch.no_grad():
                feature = encoder(reference_feature, target_rgb).squeeze(0).permute(1, 2, 0)
            feature = feature.unsqueeze(0).permute(0,3,1,2).to(accelerator.device)

            # 删除 pose 键，以保持原 batch 格式
            test_batch = accelerate.utils.send_to_device(test_batch, accelerator.device)
            del test_batch['pose']  # 删除 'pose' 键
            # 渲染一个指定位姿的图像
            rendered_output = models.render_image(
                model=model,
                accelerator=accelerator,
                batch=test_batch,
                rand=False,
                train_frac=train_frac,
                config=config
            )
            rendered_image = rendered_output["rgb"]
            # 1. 转换为 [3, 520, 780] 的形状（通道维度在前）
            rendered_image = rendered_image.permute(2, 0, 1)  # 形状变为 [3, 520, 780]
            # 2. 调整图像大小为 [3, 512, 512]
            rendered_image = F.interpolate(rendered_image.unsqueeze(0), size=(512, 512), mode='bilinear',
                                           align_corners=False).squeeze(0)
            # 3. 添加批次维度，转换为 [1, 3, 512, 512]
            rendered_image = rendered_image.unsqueeze(0).to(accelerator.device)  # 形状变为 [1, 3, 512, 512]

            text = ""
            txt = diffusion_model.text_embedder(text)
            txt = txt.to(dtype=torch.float32)
            xc_encoded = diffusion_model.image_embedder(resized_images)  # xc_encoded 维度为 [batch_size, 1280]
            xc_encoded = xc_encoded.unsqueeze(0)
            c = torch.cat([txt, xc_encoded], dim=1)
            c = diffusion_model.cross_attention_embedding(c, txt.squeeze(0))
            encoder_posterior = diffusion_model.encode_first_stage(rendered_image)
            z = diffusion_model.get_first_stage_encoding(encoder_posterior).detach()
            # 随机采样噪声级别 t
            t = torch.randint(low=tmin, high=tmax, size=(z.size(0),), dtype=torch.long, device=z.device)
            # 添加高斯噪声
            noise = torch.randn_like(z)
            # 使用 q_sample 方法生成加噪后的图像
            z_noisy = diffusion_model.q_sample(x_start=z, t=t, noise=noise)
            t_int = t.item()
            shape = [4, 512 // 8, 512 // 8]
            with torch.no_grad():
                with diffusion_model.ema_scope():
                    samples_ddim = sampler.denoising(
                        pixelnerf_features=feature,
                        cond=c,
                        shape=shape,
                        batch_size=diff_batchsize,
                        x_T=z_noisy,
                        start_T=t_int
                    )
            x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)
            x_samples_ddim = x_samples_ddim.to(accelerator.device)


            losses = {}


            sample_loss = train_utils.compute_sample_loss(rendered_image, x_samples_ddim, t)
            losses['sample'] = sample_loss
            # supervised by data
            data_loss, stats = train_utils.compute_data_loss(batch, renderings, config)
            losses['data'] = data_loss

            # interlevel loss in MipNeRF360
            if config.interlevel_loss_mult > 0 and not module.single_mlp:
                losses['interlevel'] = train_utils.interlevel_loss(ray_history, config)

            # interlevel loss in ZipNeRF360
            if config.anti_interlevel_loss_mult > 0 and not module.single_mlp:
                losses['anti_interlevel'] = train_utils.anti_interlevel_loss(ray_history, config)

            # distortion loss
            if config.distortion_loss_mult > 0:
                losses['distortion'] = train_utils.distortion_loss(ray_history, config)

            # opacity loss
            if config.opacity_loss_mult > 0:
                losses['opacity'] = train_utils.opacity_loss(renderings, config)

            # orientation loss in RefNeRF
            if (config.orientation_coarse_loss_mult > 0 or
                    config.orientation_loss_mult > 0):
                losses['orientation'] = train_utils.orientation_loss(batch, module, ray_history,
                                                                     config)
            # hash grid l2 weight decay
            if config.hash_decay_mults > 0:
                losses['hash_decay'] = train_utils.hash_decay_loss(ray_history, config)

            # normal supervision loss in RefNeRF
            if (config.predicted_normal_coarse_loss_mult > 0 or
                    config.predicted_normal_loss_mult > 0):
                losses['predicted_normals'] = train_utils.predicted_normal_loss(
                    module, ray_history, config)
            loss = sum(losses.values())
            stats['loss'] = loss.item()
            stats['losses'] = tree_map(lambda x: x.item(), losses)

            # accelerator automatically handle the scale
            accelerator.backward(loss)
            # clip gradient by max/norm/nan
            train_utils.clip_gradients(model, accelerator, config)
            optimizer.step()

            stats['psnrs'] = image.mse_to_psnr(stats['mses'])
            stats['psnr'] = stats['psnrs'][-1]

            # Log training summaries. This is put behind a host_id check because in
            # multi-host evaluation, all hosts need to run inference even though we
            # only use host 0 to record results.
            if accelerator.is_main_process:
                stats_buffer.append(stats)
                if step == init_step + 1 or step % config.print_every == 0:
                    elapsed_time = time.time() - train_start_time
                    steps_per_sec = config.print_every / elapsed_time
                    rays_per_sec = config.batch_size * steps_per_sec

                    # A robust approximation of total training time, in case of pre-emption.
                    total_time += int(round(TIME_PRECISION * elapsed_time))
                    total_steps += config.print_every
                    approx_total_time = int(round(step * total_time / total_steps))

                    # Transpose and stack stats_buffer along axis 0.
                    fs = [utils.flatten_dict(s, sep='/') for s in stats_buffer]
                    stats_stacked = {k: np.stack([f[k] for f in fs]) for k in fs[0].keys()}

                    # Split every statistic that isn't a vector into a set of statistics.
                    stats_split = {}
                    for k, v in stats_stacked.items():
                        if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
                            raise ValueError('statistics must be of size [n], or [n, k].')
                        if v.ndim == 1:
                            stats_split[k] = v
                        elif v.ndim == 2:
                            for i, vi in enumerate(tuple(v.T)):
                                stats_split[f'{k}/{i}'] = vi

                    # Summarize the entire histogram of each statistic.
                    # for k, v in stats_split.items():
                        # summary_writer.add_histogram('train_' + k, v, step)

                    # Take the mean and max of each statistic since the last summary.
                    avg_stats = {k: np.mean(v) for k, v in stats_split.items()}
                    max_stats = {k: np.max(v) for k, v in stats_split.items()}

                    summ_fn = lambda s, v: summary_writer.add_scalar(s, v, step)  # pylint:disable=cell-var-from-loop

                    # Summarize the mean and max of each statistic.
                    for k, v in avg_stats.items():
                        summ_fn(f'train_avg_{k}', v)
                    for k, v in max_stats.items():
                        summ_fn(f'train_max_{k}', v)

                    summ_fn('train_num_params', num_params)
                    summ_fn('train_learning_rate', learning_rate)
                    summ_fn('train_steps_per_sec', steps_per_sec)
                    summ_fn('train_rays_per_sec', rays_per_sec)

                    summary_writer.add_scalar('train_avg_psnr_timed', avg_stats['psnr'],
                                              total_time // TIME_PRECISION)
                    summary_writer.add_scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                                              approx_total_time // TIME_PRECISION)

                    if dataset.metadata is not None and module.learned_exposure_scaling:
                        scalings = module.exposure_scaling_offsets.weight
                        num_shutter_speeds = dataset.metadata['unique_shutters'].shape[0]
                        for i_s in range(num_shutter_speeds):
                            for j_s, value in enumerate(scalings[i_s]):
                                summary_name = f'exposure/scaling_{i_s}_{j_s}'
                                summary_writer.add_scalar(summary_name, value, step)

                    precision = int(np.ceil(np.log10(config.max_steps))) + 1
                    avg_loss = avg_stats['loss']
                    avg_psnr = avg_stats['psnr']
                    str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
                        k[7:11]: (f'{v:0.5f}' if 1e-4 <= v < 10 else f'{v:0.1e}')
                        for k, v in avg_stats.items()
                        if k.startswith('losses/')
                    }
                    logger.info(f'{step}' + f'/{config.max_steps:d}:' +
                                f'loss={avg_loss:0.5f},' + f'psnr={avg_psnr:.3f},' +
                                f'lr={learning_rate:0.2e} | ' +
                                ','.join([f'{k}={s}' for k, s in str_losses.items()]) +
                                f',{rays_per_sec:0.0f} r/s')

                    # Reset everything we are tracking between summarizations.
                    reset_stats = True

                if step > 0 and step % config.checkpoint_every == 0 and accelerator.is_main_process:
                    checkpoints.save_checkpoint(config.checkpoint_dir,
                                                accelerator, step,
                                                config.checkpoints_total_limit)

            # Test-set evaluation.
            if config.train_render_every > 0 and step % config.train_render_every == 0:
                # We reuse the same random number generator from the optimization step
                # here on purpose so that the visualization matches what happened in
                # training.
                eval_start_time = time.time()
                try:
                    test_batch = next(test_dataiter)
                except StopIteration:
                    test_dataiter = iter(test_dataloader)
                    test_batch = next(test_dataiter)
                test_batch = accelerate.utils.send_to_device(test_batch, accelerator.device)
                # 删除 pose 键，以保持原 batch 格式
                del test_batch['pose']  # 删除 'pose' 键

                # render a single image with all distributed processes
                rendering = models.render_image(model, accelerator,
                                                test_batch, False,
                                                train_frac, config)

                # move to numpy
                rendering = tree_map(lambda x: x.detach().cpu().numpy(), rendering)
                test_batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, test_batch)
                # Log eval summaries on host 0.
                if accelerator.is_main_process:
                    eval_time = time.time() - eval_start_time
                    num_rays = np.prod(test_batch['directions'].shape[:-1])
                    rays_per_sec = num_rays / eval_time
                    summary_writer.add_scalar('test_rays_per_sec', rays_per_sec, step)

                    metric_start_time = time.time()
                    metric = metric_harness(
                        postprocess_fn(rendering['rgb']), postprocess_fn(test_batch['rgb']))
                    logger.info(f'Eval {step}: {eval_time:0.3f}s, {rays_per_sec:0.0f} rays/sec')
                    logger.info(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
                    for name, val in metric.items():
                        if not np.isnan(val):
                            logger.info(f'{name} = {val:.4f}')
                            summary_writer.add_scalar('train_metrics/' + name, val, step)

                    if config.vis_decimate > 1:
                        d = config.vis_decimate
                        decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
                    else:
                        decimate_fn = lambda x: x
                    rendering = tree_map(decimate_fn, rendering)
                    test_batch = tree_map(decimate_fn, test_batch)
                    vis_start_time = time.time()
                    vis_suite = vis.visualize_suite(rendering, test_batch)
                    with tqdm.external_write_mode():
                        logger.info(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
                    if config.rawnerf_mode:
                        # Unprocess raw output.
                        vis_suite['color_raw'] = rendering['rgb']
                        # Autoexposed colors.
                        vis_suite['color_auto'] = postprocess_fn(rendering['rgb'], None)
                        summary_writer.add_image('test_true_auto',
                                                 tb_process_fn(postprocess_fn(test_batch['rgb'], None)), step)
                        # Exposure sweep colors.
                        exposures = test_dataset.metadata['exposure_levels']
                        for p, x in list(exposures.items()):
                            vis_suite[f'color/{p}'] = postprocess_fn(rendering['rgb'], x)
                            summary_writer.add_image(f'test_true_color/{p}',
                                                     tb_process_fn(postprocess_fn(test_batch['rgb'], x)), step)
                    summary_writer.add_image('test_true_color', tb_process_fn(test_batch['rgb']), step)
                    if config.compute_normal_metrics:
                        summary_writer.add_image('test_true_normals',
                                                 tb_process_fn(test_batch['normals']) / 2. + 0.5, step)
                    for k, v in vis_suite.items():
                        summary_writer.add_image('test_output_' + k, tb_process_fn(v), step)

    if accelerator.is_main_process and config.max_steps > init_step:
        logger.info('Saving last checkpoint at step {} to {}'.format(step, config.checkpoint_dir))
        checkpoints.save_checkpoint(config.checkpoint_dir,
                                    accelerator, step,
                                    config.checkpoints_total_limit)
    logger.info('Finish training.')


if __name__ == '__main__':
    with gin.config_scope('train'):
        app.run(main)
