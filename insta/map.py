import torch
import torch.nn.functional as F


def tbn(triangles):
    a, b, c = triangles.unbind(-2)
    n = F.normalize(torch.cross(b - a, c - a), dim=-1)
    d = b - a

    X = F.normalize(torch.cross(d, n), dim=-1)
    Y = F.normalize(torch.cross(d, X), dim=-1)
    Z = F.normalize(d, dim=-1)

    return torch.stack([X, Y, Z], dim=3)


def triangle2projection(triangles):
    R = tbn(triangles)
    T = triangles.unbind(-2)[0]
    I = torch.repeat_interleave(torch.eye(4, device=triangles.device)[None, None, ...], R.shape[1], 1)

    I[:, :, 0:3, 0:3] = R
    I[:, :, 0:3, 3] = T

    return I


def calculate_centroid(tris, dim=2):
    c = tris.sum(dim) / 3
    return c


def interpolate(xyzs, tris, neighbours, edge_mask):
    # Currently disabled
    return triangle2projection(tris)[0]
    N = xyzs.shape[0]
    factor = 4
    c_closests = calculate_centroid(tris)
    c_neighbours = calculate_centroid(neighbours)
    dc = torch.exp(-1 * torch.norm(xyzs - c_closests[0], dim=1, keepdim=True))
    dn = torch.exp(-factor * torch.norm(xyzs.repeat(1, 3).reshape(N * 3, 3) - c_neighbours[0], dim=1, keepdim=True)) * edge_mask
    distances = torch.cat([dc, dn.reshape(N, -1)], dim=1)
    triangles = torch.cat([triangle2projection(tris)[0][:, None, ...], triangle2projection(neighbours)[0][:, None, ...].reshape(N, -1, 4, 4)], dim=1)
    normalization = distances.sum(-1, keepdim=True)
    weights = distances / normalization

    return (triangles * weights[..., None, None]).sum(1)


def project_position(xyzs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt = interpolate(xyzs, deformed_triangles, deformed_neighbours, edge_mask)
    Rt_def = torch.linalg.inv(Rt)
    Rt_canon = interpolate(xyzs, canonical_triangles, canonical_neighbours, edge_mask)

    homo = torch.cat([xyzs, torch.ones_like(xyzs)[:, 0:1]], dim=1)[:, :, None]

    def_local = torch.matmul(Rt_def, homo)
    def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, 0:3, 0].float()


def project_direction(dirs, deformed_triangles, canonical_triangles, deformed_neighbours, canonical_neighbours, edge_mask):
    Rt = interpolate(dirs, deformed_triangles, deformed_neighbours, edge_mask)
    Rt_def = torch.linalg.inv(Rt)[:, 0:3, 0:3]
    Rt_canon = interpolate(dirs, canonical_triangles, canonical_neighbours, edge_mask)[:, 0:3, 0:3]

    def_local = torch.matmul(Rt_def, dirs[:, :, None])
    def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, 0:3, 0].float()
