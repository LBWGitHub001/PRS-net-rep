import torch
from numpy.f2py.auxfuncs import throw_error
from torch import nn
import numpy as np
from torch.nn import functional as F


class SymmetryLoss(nn.Module):
    def __init__(self):
        super(SymmetryLoss, self).__init__()

    # plane (B,3,4) quaternion(B,3,4) points(N,3)*B
    def forward(self, plane, quaternion, points_batch, nearest_idx_maps, nearest_pts_maps, max_min):
        n, d = self.convertToDirection(plane)
        if isinstance(points_batch, list):  # 全部读取(debug)
            reflect_points = list()
            rotated_points = list()
            if len(points_batch) != plane.shape[0] or len(points_batch) != quaternion.shape[0]:
                throw_error("输入的维度不对")
            for idx, points in enumerate(points_batch):
                points = points.unsqueeze(0)
                reflect_p = self.genReflectPoints(n[idx:idx + 1], d[idx:idx + 1], points)
                rotation_p = self.genRotationPoints(quaternion[idx:idx + 1], points)
                reflect_points.append(reflect_p)
                rotated_points.append(rotation_p)
            return reflect_points, rotated_points
        elif isinstance(points_batch, torch.Tensor):  # 定长采样
            reflect_p = self.genReflectPoints(n, d, points_batch)
            rotation_p = self.genRotationPoints(quaternion, points_batch)
            Dref = self.chamfer_distance(reflect_p, points_batch)
            Drot = self.chamfer_distance(rotation_p, points_batch)
            return Dref + Drot

    def convertToDirection(self, plane):
        return plane[..., 0:3], plane[..., 3]

    def genReflectPoints(self, n, d, points):
        dot = (points.unsqueeze(-2) * n.unsqueeze(1)).sum(dim=-1) + d.unsqueeze(-2)

        norm_sq = torch.sum(n * n, dim=-1).unsqueeze(-2)  # (*)

        # P' = P - 2 * dot / ‖n‖² * n
        scale = 2 * dot / norm_sq.clamp(min=1e-8)  # 广播

        # 扩展 scale 和 n 维度与 points 对齐
        reflectPoints = points.unsqueeze(-2) - scale.unsqueeze(-1) * n.unsqueeze(1)
        return reflectPoints

    def rotate_vector_by_quaternion(self, quat, vec):
        # quat: [*,4] (w,x,y,z), vec:[*,3]
        q_w, q_x, q_y, q_z = torch.unbind(quat, dim=-1)
        v_x, v_y, v_z = torch.unbind(vec, dim=-1)
        # 四元数旋转展开公式
        out_x = (1 - 2 * q_y ** 2 - 2 * q_z ** 2) * v_x + 2 * (q_x * q_y - q_w * q_z) * v_y + 2 * (
                q_x * q_z + q_w * q_y) * v_z
        out_y = 2 * (q_x * q_y + q_w * q_z) * v_x + (1 - 2 * q_x ** 2 - 2 * q_z ** 2) * v_y + 2 * (
                q_y * q_z - q_w * q_x) * v_z
        out_z = 2 * (q_x * q_z - q_w * q_y) * v_x + 2 * (q_y * q_z + q_w * q_x) * v_y + (
                1 - 2 * q_x ** 2 - 2 * q_y ** 2) * v_z
        return torch.stack([out_x, out_y, out_z], dim=-1)

    # quaternion B*3*4
    def genRotationPoints(self, Bquaternion, points):
        B, J, _ = Bquaternion.shape
        N, _ = points.shape[-2:]
        quats = Bquaternion.unsqueeze(1)  # [B,1,J,4]
        vecs = points.unsqueeze(-2)  # [B,N,1,3]
        rotated = self.rotate_vector_by_quaternion(quats, vecs)  # [B,N,J,3]
        return rotated

    def calc_distances(self, transformed_points, idx_maps, nearest_points_maps, max_min):
        # 体素化处理
        bbox_max, bbox_min = max_min
        bbox_max = bbox_max[:, None, None, :]  # (B, 1, 1, 3)
        bbox_min = bbox_min[:, None, None, :]
        size = bbox_max - bbox_min
        V = 32
        norm = (transformed_points - bbox_min) / size * V
        voxel_ijk = torch.floor(norm).clamp(0, V - 1).long()  # (B, N, M, 3)

        # 最近点获取
        B, N, M, _ = voxel_ijk.shape
        b_idx = torch.arange(B, device=voxel_ijk.device)[:, None, None]  # (B, 1, 1)
        i = voxel_ijk[..., 0].long()  # (B, N, M)
        j = voxel_ijk[..., 1].long()
        k = voxel_ijk[..., 2].long()

        # 一次性取出所有最近点 (B, N, M, 3)
        nearest_points = nearest_points_maps[b_idx, i, j, k]

        # 距离 (B, N, M)
        distances = ((transformed_points - nearest_points) ** 2).sum(dim=-1).sqrt()

        avg_distance = distances.sum(dim=1)  # (B, M)  对所有点求和
        loss = avg_distance.sum()  # scalar  所有generator累加

        return loss / B

    def chamfer_distance(self, transformed, original):
        """
        transformed: (B, N, M, 3)  变换后的点
        original:   (B, N, 3)      原始点云
        返回双向 Chamfer distance 的均值
        """
        B, N, M, _ = transformed.shape

        total_loss = 0.0
        for m in range(M):
            t = transformed[:, :, m, :]  # (B, N, 3)

            # Forward:  变换点 → 原始点 的最近距离
            d_fwd = torch.cdist(t, original)  # (B, N, N)
            loss_fwd = d_fwd.min(dim=-1)[0].mean()  # 对每个变换点取最近，再平均

            # Backward: 原始点 → 变换点 的最近距离
            d_bwd = torch.cdist(original, t)  # (B, N, N)
            loss_bwd = d_bwd.min(dim=-1)[0].mean()

            total_loss += (loss_fwd + loss_bwd)

        return total_loss / M  # 归一化到每个操作


class RegularLoss(nn.Module):
    def __init__(self):
        super(RegularLoss, self).__init__()

    def forward(self, plane, quat):
        dirs = plane[..., 0:3]
        M1 = self.convertDirToMatrix(dirs)
        M2 = self.convertDirToMatrix(quat)
        I1 = torch.eye(3, device=M1.device, dtype=M1.dtype)
        I2 = torch.eye(3, device=M1.device, dtype=M1.dtype)
        A = M1 @ M1.transpose(-1, -2) - I1
        B = M2 @ M2.transpose(-1, -2) - I2
        Dr = A**2 + B**2
        Dr = Dr.sum()
        return Dr

    def convertDirToMatrix(self, dirs):
        norm_dirs = F.normalize(dirs, dim=-1)
        return norm_dirs

    def covertQuatToNa(self, quat):
        axes = quat[..., 1:]  # (B, 3, 3)
        axes = F.normalize(axes, dim=-1)  # 单位化
        M_flat = axes.reshape_as(axes)  # (B, 9)
        return M_flat


class PRSLoss(nn.Module):
    def __init__(self):
        super(PRSLoss, self).__init__()
        self.Symmetry = SymmetryLoss()
        self.Regular = RegularLoss()
        self.gamma = 0.8

    def forward(self, planes, quaternions, points_batch, nearest_idx_maps, nearest_pts_maps, max_min):
        Ds = self.Symmetry(planes, quaternions, points_batch, nearest_idx_maps, nearest_pts_maps, max_min)
        Dr = self.Regular(planes, quaternions)
        return Ds + self.gamma * Dr
    # ===================== 测试 =====================


import torch
import numpy as np
from PRSLoss import SymmetryLoss
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.manual_seed(42)
    Loss = SymmetryLoss()

    # ===================== 构造输入 =====================
    B = 2  # 对称生成器数量
    M = 3  # 每个生成器的对称操作数

    plane = torch.randn(B, M, 4)  # (B, M, 4)
    quat = torch.randn(B, M, 4)  # (B, M, 4)
    quat = torch.nn.functional.normalize(quat, dim=-1)  # 单位四元数

    # 变长点云：3 个样本，点数各不相同
    points_batch = [
        torch.randn(80, 3),  # 80 个点
        torch.randn(120, 3),  # 120 个点
    ]

    print("=" * 60)
    print("输入信息")
    print("=" * 60)
    print(f"  plane 形状:      {list(plane.shape)}")
    print(f"  quat 形状:       {list(quat.shape)}")
    print(f"  points_batch 长度: {len(points_batch)}")
    for i, pts in enumerate(points_batch):
        print(f"    样本 {i}: {list(pts.shape)}")

    # ===================== 调用 forward =====================
    reflect_list, rotate_list = Loss(plane, quat, points_batch)

    # ===================== 验证输出 =====================
    print("\n" + "=" * 60)
    print("输出验证")
    print("=" * 60)
    print(f"  reflect_list 长度: {len(reflect_list)}")
    print(f"  rotate_list  长度: {len(rotate_list)}")

    all_ok = True
    for i, (ref, rot) in enumerate(zip(reflect_list, rotate_list)):
        N_i = points_batch[i].shape[0]
        expected_shape = (1, N_i, M, 3)

        ok_ref = list(ref.shape) == list(expected_shape)
        ok_rot = list(rot.shape) == list(expected_shape)

        status_ref = "✅" if ok_ref else "❌"
        status_rot = "✅" if ok_rot else "❌"

        print(f"\n  样本 {i} (原始点数={N_i}):")
        print(f"    {status_ref} reflect: {list(ref.shape)}  期望: {list(expected_shape)}")
        print(f"    {status_rot} rotate:  {list(rot.shape)}  期望: {list(expected_shape)}")

        if not ok_ref or not ok_rot:
            all_ok = False

    print("\n" + "=" * 60)
    print("🏁 全部通过" if all_ok else "❌ 存在形状不匹配")
    print("=" * 60)
