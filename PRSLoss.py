import torch
from torch import nn
import numpy as np
import quaternion


class SymmetryLoss(nn.Module):
    def __init__(self):
        super(SymmetryLoss, self).__init__()

    def forward(self, plane, quaternion, points):
        n, d = self.convertToDirection(plane)
        reflect_p = self.genReflectPoints(n, d, points)
        rotation_p = self.genRotationPoints(quaternion, points)

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
            p = quaternion.as_quat_array(quat)
            q = quaternion.from_vector_part(points)
            q_ = p * q * p.conjugate()
            rotated_list.append(quaternion.as_vector_part(q_))

        rotated_points = torch.tensor(rotated_list)
        return rotated_points.reshape_as(points)


if __name__ == "__main__":
    Loss = SymmetryLoss()
    # ── 测试数据 ──
    n = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])  # (1, 3, 3)
    d = torch.tensor([[-1., -2., -3.]])  # (1, 3)
    points = torch.tensor([[[0., 0., 0.], [1., 2., 3.]]])  # (1, 2, 3)

    # ── 向量化计算 ──
    reflect_vec = Loss.genReflectPoints(n, d, points)

    # ── 手动循环计算 ──
    B, M, _ = n.shape
    _, N, _ = points.shape
    reflect_manual = torch.zeros(B, N, M, 3)
    for b in range(B):
        for k in range(N):
            for i in range(M):
                ni, di, pk = n[b, i], d[b, i], points[b, k]
                t = 2 * (ni @ pk + di) / (ni @ ni)
                reflect_manual[b, k, i] = pk - t * ni

    # ── 比对 ──
    diff = (reflect_vec - reflect_manual).abs().max().item()
    print(f"shape: {reflect_vec.shape}  vs  {reflect_manual.shape}")
    print(f"max diff: {diff:.2e}")
    print("✅ 一致" if diff < 1e-5 else "❌ 不一致")

    # ===================== 四元数测试开始 =====================
    # 1. 构造输入
    Bquat = torch.randn(1, 1, 4)  # 最简单形状 [B=1, 1个四元数, 4]
    Bquat = torch.nn.functional.normalize(Bquat, dim=-1)  # 单位化

    pts = torch.tensor([[[1., 0., 0.]]])  # 点 [1,0,0]

    # 2. 你的函数结果
    out1 = Loss.genRotationPoints(Bquat, pts)

    # 3. 手动四元数旋转 (p * v * p*)
    p = quaternion.as_quat_array(Bquat.numpy())
    v = quaternion.from_vector_part(pts.numpy())
    out2 = torch.tensor(quaternion.as_vector_part(p * v * p.conj()))

    # 4. 对比
    print("你的函数输出：", out1)
    print("手动计算输出：", out2)
    print("是否一致：", torch.allclose(out1, out2, atol=1e-6))
