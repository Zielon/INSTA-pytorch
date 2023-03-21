import numpy as np
import trimesh


def parse_mask(mesh):
    mask = np.array(mesh.visual.face_colors[:, 0:3])
    mask = mask[:, 0] > 240
    ids = np.arange(0, len(mask))[mask]
    return ids.tolist()


class MeshUtils():
    def __init__(self):
        self.lips_ids = parse_mask(trimesh.load('/home/wzielonka/CLionProjects/real-time-avatar/mesh_masks/lips.obj', process=False))
