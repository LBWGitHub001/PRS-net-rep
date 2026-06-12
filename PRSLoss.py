import torch
from numpy.f2py.auxfuncs import throw_error
from torch import nn
import numpy as np
import quaternion


class SymmetryLoss(nn.Module):
    def __init__(self):
        super(SymmetryLoss, self).__init__()

    # plane (B,3,4) quaternion(B,3,4) points(N,3)*B
    def forward(self, plane, quaternion, points_batch):
        n, d = self.convertToDirection(plane)
        if isinstance(points_batch, list): # 全部读取
            reflect_points = list()
            rotated_points = list()
            if len(points_batch) != plane.shape[0] or len(points_batch) != quaternion.shape[0]:
                throw_error("输入的维度不对")
            for idx, points in enumerate(points_batch):
                points = points.unsqueeze(0)
                reflect_p = self.genReflectPoints(n[idx:idx+1], d[idx:idx+1], points)
                rotation_p = self.genRotationPoints(quaternion[idx:idx+1], points)
                reflect_points.append(reflect_p)
                rotated_points.append(rotation_p)
            return reflect_points, rotated_points
        elif isinstance(points_batch, torch.Tensor): # 定长采样
            reflect_p = self.genReflectPoints(n, d, points_batch)
            rotation_p = self.genRotationPoints(quaternion, points_batch)


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

    # quaternion B*3*4
    def genRotationPoints(self, Bquaternion, points):
        Lquaternions = torch.unbind(Bquaternion, dim=1)
        rotated_list = []
        for quat in Lquaternions:
            p = quaternion.as_quat_array(quat.detach().numpy())
            p = p[:, np.newaxis]
            q = quaternion.from_vector_part(points)
            q_ = p * q * p.conjugate()
            rotated_list.append(torch.tensor(quaternion.as_vector_part(q_)))

        rotated_points = torch.stack(rotated_list,dim=-2)
        return rotated_points


# ===================== 测试 =====================
import torch
import numpy as np
from PRSLoss import SymmetryLoss
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.manual_seed(42)
    Loss = SymmetryLoss()

    # ===================== 构造输入 =====================
    B = 2        # 对称生成器数量
    M = 3        # 每个生成器的对称操作数

    plane = torch.randn(B, M, 4)                           # (B, M, 4)
    quat = torch.randn(B, M, 4)                            # (B, M, 4)
    quat = torch.nn.functional.normalize(quat, dim=-1)     # 单位四元数

    # 变长点云：3 个样本，点数各不相同
    points_batch = [
        torch.randn(80, 3),    # 80 个点
        torch.randn(120, 3),   # 120 个点
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



