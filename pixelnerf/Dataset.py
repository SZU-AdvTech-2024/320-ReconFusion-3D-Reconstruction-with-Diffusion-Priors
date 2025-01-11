import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from pixelnerf.ImageEncoder import ImageEncoder
from pixelnerf import utils, camera_utils
from pixelnerf.pycolmap import pycolmap
from tqdm import tqdm


def sample_rays_np(H, W, fx, fy, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5 + 0.5) / fx, -(j - H * .5 + 0.5) / fy, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_dataset(data_dir, n, device):
    data = np.load(data_dir)
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    train_list = shuffle_id(images.shape[0], n)
    H, W = images.shape[1:3]
    rays = create_ray_batches(images, poses, train_list, H, W, focal, device)

    train_images = torch.tensor(images[train_list], device=device).permute(0, 3, 1, 2)
    encoder = ImageEncoder().to(device).eval()
    with torch.no_grad():
        reference_feature = encoder(train_images)
        # reference_feature => tensor(n, 512, 50, 50)

    return RaysDataset(rays), ReferenceDataset(reference_feature, poses[train_list], focal, H)

def get_data(data_dir, device, factor, mode):
    # Set up scaling factor.
    image_dir_suffix = ''
    # Use downsampling factor (unless loading training split for raw dataset,
    # we train raw at full resolution because of the Bayer mosaic pattern).
    if factor > 0:
        image_dir_suffix = f'_{factor}'
        factor = factor
    else:
        factor = 1

    colmap_dir = os.path.join(data_dir, 'sparse/0')

    # Load poses.
    if utils.file_exists(colmap_dir):
        pose_data = NeRFSceneManager(colmap_dir).process()
    else:
        raise ValueError('Colmap data not found.')
    image_names, poses, pixtocam, distortion_params, camtype, fx, fy = pose_data

    # Load bounds if possible (only used in forward facing scenes).
    posefile = os.path.join(data_dir, 'poses_bounds.npy')
    if utils.file_exists(posefile):
        with utils.open_file(posefile, 'rb') as fp:
            poses_arr = np.load(fp)
        bounds = poses_arr[:, -2:]
    else:
        bounds = np.array([0.01, 1.])

    # Recenter poses.
    poses, transform = camera_utils.recenter_poses(poses)
    poses = poses.astype(np.float32)
    # Load images.
    colmap_image_dir = os.path.join(data_dir, 'images')
    image_dir = os.path.join(data_dir, 'images' + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
        if not utils.file_exists(d):
            raise ValueError(f'Image folder {d} does not exist.')
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(utils.listdir(colmap_image_dir))
    image_files = sorted(utils.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [os.path.join(image_dir, colmap_to_image[f])
                   for f in image_names]
    images = [utils.load_img(x) for x in tqdm(image_paths, desc='Loading LLFF dataset', leave=False)]
    images = np.stack(images, axis=0) / 255.

    if mode == 'train':
        n = 3
        train_list = shuffle_id(images.shape[0], n)
        H, W = images.shape[1:3]
        rays = create_ray_batches(images, poses, train_list, H, W, fx, fy, device)
        rays = rays.float()
        train_images = torch.tensor(images[train_list], device=device).permute(0, 3, 1, 2)
        encoder = ImageEncoder().to(device).eval()
        with torch.no_grad():
            reference_feature = encoder(train_images)
            # reference_feature => tensor(n, 512, 50, 50)

        return RaysDataset(rays), ReferenceDataset(reference_feature, poses[train_list], fx, fy, H), bounds
    else:
        n = 4
        list = shuffle_id(images.shape[0], n)
        train_list = list[:3]
        test_list = list[-1]
        H, W = images.shape[1:3]
        rays = create_ray_batches(images, poses, train_list, H, W, fx, fy, device)
        rays = rays.float()
        train_images = torch.tensor(images[train_list], device=device).permute(0, 3, 1, 2)
        target_image = torch.tensor(images[test_list], device=device).unsqueeze(0).permute(0, 3, 1, 2)
        target_pose = poses[test_list]
        encoder = ImageEncoder().to(device).eval()
        with torch.no_grad():
            reference_feature = encoder(train_images)
            # reference_feature => tensor(n, 512, 50, 50)

        return RaysDataset(rays), ReferenceDataset(reference_feature, poses[train_list], fx, fy, H), bounds, target_pose, target_image, train_images

def shuffle_id(n, k):
    train_list = np.arange(n)
    np.random.shuffle(train_list)
    train_list = train_list[:k]
    return train_list


def create_ray_batches(images, poses, train_list, H, W, fx, fy, device):
    print("Create Ray batches!")
    rays_o_list = list()
    rays_d_list = list()
    rays_rgb_list = list()
    for i in train_list:
        img = images[i]
        pose = poses[i]
        rays_o, rays_d = sample_rays_np(H, W, fx, fy, pose)
        rays_o_list.append(rays_o.reshape(-1, 3))
        rays_d_list.append(rays_d.reshape(-1, 3))
        rays_rgb_list.append(img.reshape(-1, 3))
    rays_o_npy = np.concatenate(rays_o_list, axis=0)
    rays_d_npy = np.concatenate(rays_d_list, axis=0)
    rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
    rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=device)
    return rays


class RaysDataset(Dataset):
    def __init__(self, rays):
        self.rays = rays

    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, idx):
        return self.rays[idx]


class ReferenceDataset:
    def __init__(self, reference, c2w, fx, fy, img_size):
        self.reference = reference
        self.scale = (img_size / 2) / (fx + fy) / 2
        self.n = c2w.shape[0]
        self.R_t = torch.tensor(c2w[:, :3, :3], device=reference.device).permute(0, 2, 1)
        self.camera_pos = torch.tensor(c2w[:, :3, -1], device=reference.device)
        self.c2w = c2w
        self.img_size = img_size
        self.fx = fx
        self.fy = fy

    @torch.no_grad()
    def feature_matching(self, pos):
        n_rays, n_samples, _ = pos.shape
        pos = pos.unsqueeze(dim=0).expand([self.n, n_rays, n_samples, 3])
        camera_pos = self.camera_pos[:, None, None, :]
        camera_pos = camera_pos.expand_as(pos)
        ref_pos = torch.einsum("kij,kbsj->kbsi", self.R_t, pos-camera_pos)
        uv_pos = ref_pos[..., :-1] / ref_pos[..., -1:] / self.scale
        uv_pos[..., 1] *= -1.0
        # uv_pos = uv_pos.float()
        return F.grid_sample(self.reference, uv_pos, align_corners=True, padding_mode="border")


class NeRFSceneManager(pycolmap.SceneManager):
    """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """

    def process(self):
        """Applies NeRF-specific postprocessing to the loaded pose data.

    Returns:
      a tuple [image_names, poses, pixtocam, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
    """

        self.load_cameras()
        self.load_images()
        # self.load_points3D()  # For now, we do not need the point cloud data.

        # Assume shared intrinsics between all cameras.
        cam = self.cameras[1]

        # Extract focal lengths and principal point parameters.
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :3, :4]

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]

        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 1 or type_ == 'PINHOLE':
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 3 or type_ == 'RADIAL':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 4 or type_ == 'OPENCV':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['p1'] = cam.p1
            params['p2'] = cam.p2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
            params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
            params['k1'] = cam.k1
            params['k2'] = cam.k2
            params['k3'] = cam.k3
            params['k4'] = cam.k4
            camtype = camera_utils.ProjectionType.FISHEYE

        return names, poses, pixtocam, params, camtype, fx, fy