# ReconFusion

An unofficial reconfusion implementation of 
"ReconFusion: 3D Reconstruction with Diffusion Priors" 
[https://arxiv.org/abs/2312.02981](https://arxiv.org/abs/2312.02981).
This work is based on [zipnerf](https://github.com/SuLvXiangXin/zipnerf-pytorch), [pixelnerf](https://github.com/kunkun0w0/Clean-Torch-NeRFs), [stable-diffusion](https://github.com/CompVis/stable-diffusion).

Due to the slow speed of Nerf training and the addition of the Diffusion Model to each round of training, the checkpoints were only able to train 150 rounds with insufficient computational resources. If necessary and resources are sufficient, training can continue on this basis until convergence.

If it is necessary to train in the self-training concentration, it should be noted that the Diffusion Model should be stripped out and trained separately after getting the trained checkpoint, and then brought back into the reconfusion to train Nerf.

## CheckPoints Download
Diffusion and PixelNerf Checkpoints(1.5): [Pretrained weights](https://drive.google.com/drive/folders/1Aqd8SlxsUdwbWB-sZx_EVxi36AGchIr4?usp=drive_link)
All checkpoints folders are placed directly under the reconfusion directory! Also change the checkpoints path in train.py to point to your corresponding file path.

Zip-Nerf Checkpoints:[Pretrained weights](https://drive.google.com/drive/folders/1Fgmqt6NNW_hdH3DmIVKXU4758VFMjbsw?usp=sharing)
Place the checkpoints folder under exp/360_v2/bonsai!

CLIP Checkpoints: [weights](https://drive.google.com/drive/folders/1aoNykGDKaptwJMqyE9SCjKXlTWXWWqe5?usp=sharing)
Place the folder under the diffusion/openai folder!

VAE Checkpoints: [weights](https://drive.google.com/drive/folders/1y6HVZyDZXMAG5ex1_KsVJuDJpjctISwT?usp=sharing)
Place the folder under the diffusion/models/first_stage_models folder!

## Data Download

360_v2/bonsai:
[data](https://drive.google.com/drive/folders/1nDgr5CaWZdYVq9qypKzeOHQjF5EdRG_6?usp=sharing)
The data folder is also placed directly under the reconfusion directory!


## Install CUDA backend

```
# Clone the repo.
git clone xxxxxx.git
cd reconfusion

# Make a conda environment.
conda create --name reconfusion python=3.9
conda activate reconfusion

# Install requirements.
pip install -r requirements.txt

# Install other cuda extensions
pip install ./extensions/cuda

# Install a specific cuda version of torch_scatter 
# see more detail at https://github.com/rusty1s/pytorch_scatter
CUDA=cu117
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html


## Train
```
# Where your data is 
DATA_DIR=data/bonsai
EXP_NAME=360_v2/bonsai

# Experiment will be conducted under "exp/${EXP_NAME}" folder
# "--gin_configs=configs/360.gin" can be seen as a default config 
# and you can add specific config useing --gin_bindings="..." 
python train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
      --gin_bindings="Config.factor = 4" 

# metric, render image, etc can be viewed through tensorboard
tensorboard --logdir "exp/${EXP_NAME}"

```
### Render
Rendering results can be found in the directory `exp/${EXP_NAME}/render`
```
python render.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 480" \
    --gin_bindings="Config.render_video_fps = 60" \
    --gin_bindings="Config.factor = 4"  

## OutOfMemory
you can decrease the total batch size by 
adding e.g.  `--gin_bindings="Config.batch_size = 8192" `, 
or decrease the test chunk size by adding e.g.  `--gin_bindings="Config.render_chunk_size = 8192" `,
or use more GPU by configure `accelerate config` .


## Citation
```
@article{wu2023reconfusion,
    title={ReconFusion: 3D Reconstruction with Diffusion Priors},
    author={Rundi Wu and Ben Mildenhall and Philipp Henzler and 
			Keunhong Park and Ruiqi Gao and Daniel Watson and 
			Pratul P. Srinivasan and Dor Verbin and Jonathan T. Barron 
			and Ben Poole and Aleksander Holynski},
    journal={arXiv},
    year={2023}
	}
```

## Acknowledgements
- Thanks to [zipnerf](https://github.com/SuLvXiangXin/zipnerf-pytorch) for amazing zipnerf implementation
- Thanks to [pixelnerf](https://github.com/kunkun0w0/Clean-Torch-NeRFs) for clean pixelnerf implementation
- Thanks to [stable-diffusion](https://github.com/CompVis/stable-diffusion) for amazing stable-diffusion implementation
