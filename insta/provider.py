import glob
import json
import os

import cv2
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    # new_pose = np.array([
    #     [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
    #     [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
    #     [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
    #     [0, 0, 0, 1],
    # ], dtype=np.float32)
    return pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi / 3, 2 * np.pi / 3], phi_range=[0, 2 * np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)  # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


def get_graph(mesh):
    faces = mesh.faces
    edges = mesh.edges
    adjacency = mesh.face_adjacency
    graph = []
    for i in range(faces.shape[0]):
        graph.append(set())

    for ij in adjacency:
        a = ij[0]
        b = ij[1]
        graph[a].add(b)
        graph[b].add(a)

    topology = []
    edge_face = []
    for face in graph:
        l = list(face)
        is_edge_face = 1
        if len(l) != 3:
            l = [0, 0, 0]
            is_edge_face = 0
        topology.append(l)
        edge_face.append(is_edge_face)

    topology = np.array(topology)
    edge_face = np.array(edge_face)
    return topology, edge_face


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16  # if preload, load into fp16.

        mesh = trimesh.load(self.root_path + '/canonical.obj', process=False)

        topology, edge_mask = get_graph(mesh)

        self.topology = torch.from_numpy(topology).cuda().int()
        self.edge_face_mask = torch.from_numpy(edge_mask).cuda().int()
        self.canonical_triangles = self.get_triangles(mesh).unsqueeze(dim=0).cuda()
        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        self.cache = {}
        self.reload()

    def reload(self):
        # auto-detect transforms.json and split mode.
        # if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
        #     self.mode = 'colmap' # manually split, use view-interpolation for test.
        if os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender'  # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif self.type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{self.type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // self.downscale
            self.W = int(transform['w']) // self.downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]

        MAX_FRAMES = 1500

        if self.type == "train":
            np.random.shuffle(frames)
            frames = frames[:MAX_FRAMES]
        elif self.type == "test":
            frames = sorted(frames, key=lambda d: d['file_path'])

        MAX_FRAMES = len(frames)
        self.poses = [None] * MAX_FRAMES
        self.images = [None] * MAX_FRAMES
        self.meshes = [None] * MAX_FRAMES
        self.depth = [None] * MAX_FRAMES
        self.seg = [None] * MAX_FRAMES
        self.exp = [None] * MAX_FRAMES

        for i in tqdm(range(MAX_FRAMES)):
            f = frames[i]
            f_path = os.path.join(self.root_path, f['file_path'])
            if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                f_path += '.png'  # so silly...

            # there are non-exist paths in fox...
            if not os.path.exists(f_path):
                return

            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

            if f_path in self.cache:
                image = self.cache[f_path]
            else:
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                self.cache[f_path] = image

            if self.H is None or self.W is None:
                self.H = image.shape[0] // self.downscale
                self.W = image.shape[1] // self.downscale

            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

            image = image.astype(np.float32) / 255  # [H, W, 3/4]

            path = f_path.replace('images', 'meshes').replace('png', 'obj')
            if path in self.cache:
                mesh = self.cache[path]
            else:
                mesh = trimesh.load(path, process=False)
                self.cache[path] = mesh

            path = f_path.replace('images', 'depth')
            if path in self.cache:
                depth = self.cache[path]
            else:
                depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., None] / 1000.0
                self.cache[path] = depth

            path = f_path.replace('images', 'seg_mask')
            if path in self.cache:
                seg = self.cache[path]
            else:
                seg = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., 2:3]
                self.cache[path] = seg

            path = f_path.replace('images', '/flame/exp').replace('png', 'txt')
            if path in self.cache:
                exp = self.cache[path]
            else:
                exp = np.loadtxt(path)[:16]
                self.cache[path] = exp

            self.poses[i] = pose
            self.images[i] = image
            self.meshes[i] = mesh
            self.depth[i] = depth
            self.seg[i] = seg
            self.exp[i] = exp

        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        self.exp = torch.from_numpy(np.stack(self.exp, axis=0)).float()
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0))  # [N, H, W, C]
            self.depth = torch.from_numpy(np.stack(self.depth, axis=0))  # [N, H, W, C]
            self.seg = torch.from_numpy(np.stack(self.seg, axis=0))  # [N, H, W, C]

        triangles = []
        for mesh in self.meshes:
            triangles.append(self.get_triangles(mesh))
        self.meshes = torch.from_numpy(np.stack(triangles, axis=0))

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        # print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float)  # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            self.meshes = self.meshes.to(self.device)
            self.depth = self.depth.to(self.device)
            self.seg = self.seg.to(self.device)
            self.exp = self.exp.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def get_triangles(self, mesh):
        v = mesh.vertices
        vertices = torch.tensor(v, dtype=torch.float32)
        faces = torch.tensor(mesh.faces.astype(np.int64), dtype=torch.long)
        return vertices[faces]

    def process_mesh(self, index):
        triangles = []
        for i in index:
            input_mesh = self.meshes[i]
            triangles.append(self.get_triangles(input_mesh))
        return torch.cat(triangles, dim=0)

    def collate(self, index):

        B = len(index)  # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):
            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays)  # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
            }

        poses = self.poses[index].to(self.device)  # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]

        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'triangles': self.meshes[index].to(self.device),
            'exp': self.exp[index].to(self.device),
        }

        if self.images is not None:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            depth = self.depth[index].to(self.device)  # [B, H, W, 3/4]
            seg = self.seg[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
                depth = torch.gather(depth.view(B, -1, 1), 1, torch.stack([rays['inds']], -1))  # [B, N, 3/4]
                seg = torch.gather(seg.view(B, -1, 1), 1, torch.stack([rays['inds']], -1))  # [B, N, 3/4]

            results['images'] = images
            results['depth'] = depth
            results['seg'] = seg

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose  # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
