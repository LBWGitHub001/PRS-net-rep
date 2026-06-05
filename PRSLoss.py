import torch
from torch import nn


class SymmetryLoss(nn.Module):
    def __init__(self):
        super(SymmetryLoss, self).__init__()

    def forward(self, plane, points):
        n, d  = self.convertToDirection(plane)
        reflect_p = self.genReflectPoints(n, d, points)

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

    def antiReflectPoints(self, plane, reflectPoints):
        pass



if __name__ == "__main__":
    # Batch=2，每个模型输出 3 个平面 (a,b,c,d)
    planes_multi = torch.tensor([
        # ── 模型 0 ──
        [[0., 0., 1., -2.],  # z=2
         [0., 1., 0., -1.],  # y=1
         [1., 0., 0., -3.]],  # x=3
        # ── 模型 1 ──
        [[0., 0., 1., 0.],  # z=0
         [0., 1., 0., 0.],  # y=0
         [1., 0., 0., 0.]],  # x=0
    ])  # (2, 3, 4)

    points_multi = torch.randn(2, 1024, 3)  # (2, 1024, 3)

    Loss = SymmetryLoss()
    l = Loss(planes_multi, points_multi)

    import torch

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
