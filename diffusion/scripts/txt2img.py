import argparse, os, sys, glob

from matplotlib import pyplot as plt

sys.path.append("D:/Reconfusion/stable-diffusion")
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.models.diffusion.ddpm import LatentDiffusion, LatentDiffusionSampler
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "D:\\Reconfusion\\stable-diffusion\\CompVis\\stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
pixelnerf_path = "D:\\Reconfusion\\stable-diffusion\\saved_features\\test4\\all_pixelnerf_features.pt"
input_images_path="D:\\Reconfusion\\stable-diffusion\\saved_features\\test4\\all_input_images.pt"
target_rgb_path="D:\\Reconfusion\\stable-diffusion\\saved_features\\test4\\all_target_image.pt"

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="D:\\Reconfusion\\stable-diffusion\\logs\\2024-11-23T15-14-20_txt2img-1p4B-eval\\configs\\2024-11-23T15-14-20-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="D:\\Reconfusion\\stable-diffusion\\logs\\2024-11-23T15-14-20_txt2img-1p4B-eval\\checkpoints\\epoch=000057.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs\latent-diffusion\\txt2img-1p4B-eval.yaml"
        opt.ckpt = "models\ldm\\text2img-large\model.ckpt"
        opt.outdir = "outputs\\txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    # if not opt.from_file:
    #     prompt = opt.prompt
    #     assert prompt is not None
    #     data = [batch_size * [prompt]]
    #
    # else:
    #     print(f"reading prompts from {opt.from_file}")
    #     with open(opt.from_file, "r") as f:
    #         data = f.read().splitlines()
    #         data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    pixelnerf_features = torch.load(pixelnerf_path).to(device)  # 加载 PixelNeRF 特征
    pixelnerf_features = pixelnerf_features.permute(0,3,1,2).to(device)
    input_images = torch.load(input_images_path).to(device)  # 加载 CLIP 特征
    input_images = input_images.squeeze(0)
    text = ""
    txt = model.text_embedder(text)
    txt = txt.to(dtype=torch.float32)
    xc_encoded = model.image_embedder(input_images)  # xc_encoded 维度为 [batch_size, 1280]
    xc_encoded = xc_encoded.unsqueeze(0)
    # print(f"txt max: {txt.max()}, txt min: {txt.min()}")
    # print(f"xc_encoded max: {xc_encoded.max()}, xc_encoded min: {xc_encoded.min()}")
    c = torch.cat([txt, xc_encoded], dim=1)
    # 将拼接后的结果送入密集层（CrossAttentionEmbedding）
    c = model.cross_attention_embedding(c, txt.squeeze(0))

    target_rgb = torch.load(target_rgb_path).to(device).squeeze(0)  # 加载目标 RGB 图像数据
    encoder_posterior = model.encode_first_stage(target_rgb)
    z = model.get_first_stage_encoding(encoder_posterior).detach()


    # print("Max value:", torch.max(target_rgb).item())
    # print("Min value:", torch.min(target_rgb).item())
    # if isinstance(target_rgb, torch.Tensor):
    #     train_images = target_rgb.cpu().numpy()
    # # 如果 train_images 是 [N, C, H, W] 格式，将其转为 [N, H, W, C]
    # if train_images.shape[1] == 3 or train_images.shape[1] == 1:  # 3: RGB, 1: Gray
    #     train_images = train_images.transpose(0, 2, 3, 1)
    #
    # # 将 [-1, 1] 映射到 [0, 1]
    # # if train_images.min() < 0:
    # #     train_images = (train_images + 1.0) / 2.0
    # #     train_images = np.clip(train_images, 0, 1)  # 确保范围在 [0, 1]
    #
    # # 保存图像
    # plt.imshow(train_images[0])
    # plt.axis('off')
    # plt.title("Train Image 3")
    # # 定义保存文件的路径
    # file_path = "unknown/3.png"
    # # 获取文件所在的目录
    # directory = os.path.dirname(file_path)
    # # 如果目录不存在，则创建它
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # plt.savefig("unknown/3.png")
    # print("Image saved as 3.png")

    t = torch.full((z.size(0),), 999, dtype=torch.long, device=z.device)

    # 添加高斯噪声
    noise = torch.randn_like(z)

    # 使用 q_sample 方法生成加噪后的图像
    z_noisy = model.q_sample(x_start=z, t=t, noise=noise)
    # 打印最大值和最小值
    print("Max value:", torch.max(z_noisy).item())
    print("Min value:", torch.min(z_noisy).item())


    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    # if isinstance(prompts, tuple):
                    #     prompts = list(prompts)
                    # c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(pixelnerf_features=pixelnerf_features,
                                                     S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     eta=opt.ddim_eta,
                                                     x_T=z_noisy)
                    # samples_ddim = sampler.denoising(
                    #     pixelnerf_features=pixelnerf_features,
                    #     cond=c,
                    #     shape=shape,
                    #     batch_size=opt.n_samples,
                    #     x_T=z_noisy
                    # )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = x_samples_ddim.squeeze(0).cpu().numpy()
                    if len(x_samples_ddim.shape) == 4:  # [B, C, H, W]
                        x_samples_ddim = x_samples_ddim.transpose(0, 2, 3, 1)  # 转为 [B, H, W, C]
                    elif len(x_samples_ddim.shape) == 3:  # [C, H, W]
                        x_samples_ddim = x_samples_ddim.transpose(1, 2, 0)  # 转为 [H, W, C]
                    else:
                        raise ValueError(f"Unexpected shape for x_samples_ddim: {x_samples_ddim.shape}")

                    # 确保数值范围在 [0, 1]
                    x_samples_ddim = np.clip(x_samples_ddim, 0, 1)
                    # 转换为 8-bit 图像
                    x_samples_ddim = (x_samples_ddim * 255).astype(np.uint8)
                    path = os.path.join(sample_path, f"{base_count:05}.png")
                    Image.fromarray(x_samples_ddim).save(path)

                    # 打印最大值和最小值
                    # print("Max value:", torch.max(x_samples_ddim).item())
                    # print("Min value:", torch.min(x_samples_ddim).item())
                    # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    #
                    # # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    # # 直接跳过 NSFW 检测
                    # x_checked_image = x_samples_ddim
                    # has_nsfw_concept = [False] * len(x_samples_ddim)
                    # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    #
                    # if not opt.skip_save:
                    #     for x_sample in x_checked_image_torch:
                    #         x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    #         img = Image.fromarray(x_sample.astype(np.uint8))
                    #         # img = put_watermark(img, wm_encoder)
                    #         img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    #         base_count += 1
                    #
                    # if not opt.skip_grid:
                    #     all_samples.append(x_checked_image_torch)

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
