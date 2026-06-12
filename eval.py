import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from PRSNet import PRSNet
from Reader import CombinedDataset
from PRSLoss import SymmetryLoss


def evaluate(model, dataset, loss_func, device='cpu', num_samples=5):
    """
    推理评估：取 num_samples 个样本，计算 loss 并可视化。
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    results = []

    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            # 取数据
            voxel, points = dataset[idx]                    # (1,32,32,32), (N,3)
            near_idx, near_pts = dataset.get_nearest(idx)   # (32,32,32) int32, (32,32,32,3)
            bbox_min = torch.tensor(dataset.bbox_mins[idx]).to(device)
            bbox_max = torch.tensor(dataset.bbox_maxs[idx]).to(device)

            # 拼 batch 维度
            voxel_b = voxel.unsqueeze(0).to(device)         # (1, 1, 32, 32, 32)
            points_b = points.to(device)                     # (N, 3)
            near_idx_b = near_idx.unsqueeze(0).to(device)    # (1, 32, 32, 32)
            near_pts_b = near_pts.unsqueeze(0).to(device)    # (1, 32, 32, 32, 3)
            max_min = (bbox_max.unsqueeze(0), bbox_min.unsqueeze(0))

            # 推理
            planes, quaternions = model(voxel_b)             # (1, M, 4), (1, M, 4)

            # 计算 loss
            loss = loss_func(planes, quaternions,
                             [points_b],
                             near_idx_b, near_pts_b, max_min)

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            total_loss += loss_val

            results.append({
                'idx': idx,
                'model_id': dataset.get_model_info(idx)['model_id'],
                'category': dataset.get_model_info(idx)['category'],
                'loss': loss_val,
                'voxel': voxel_b,
                'points': points_b,
                'planes': planes,
                'quaternions': quaternions,
            })

            print(f"[eval {idx+1}/{num_samples}] "
                  f"{dataset.get_model_info(idx)['model_id'][:16]}  "
                  f"loss: {loss_val:.6f}")

    avg_loss = total_loss / num_samples
    print(f"\n{'='*50}")
    print(f"平均 Loss: {avg_loss:.6f}")
    print(f"{'='*50}")

    return results


# ===================== 可视化 =====================
def visualize_symmetry(result, dataset, sample_idx=0):
    """
    可视化某样本的对称变换效果
    result: evaluate() 返回的列表元素
    """
    loss_func = SymmetryLoss()

    voxel_b = result['voxel']
    points_b = result['points']
    planes = result['planes']
    quaternions = result['quaternions']
    idx = result['idx']

    near_pts = dataset.get_nearest(idx)[1]                 # (32,32,32,3)
    near_pts_b = near_pts.unsqueeze(0).to(voxel_b.device)
    bbox_min = torch.tensor(dataset.bbox_mins[idx]).to(voxel_b.device)
    bbox_max = torch.tensor(dataset.bbox_maxs[idx]).to(voxel_b.device)
    max_min = (bbox_max.unsqueeze(0), bbox_min.unsqueeze(0))

    with torch.no_grad():
        n, d = loss_func.convertToDirection(planes)         # (1, M, 3), (1, M)
        reflect_p = loss_func.genReflectPoints(n, d, points_b.unsqueeze(0))  # (1, N, M, 3)
        rotate_p = loss_func.genRotationPoints(quaternions, points_b.unsqueeze(0))

    M = planes.shape[1]
    N = points_b.shape[0]
    sample_n = min(N, 500)

    pts_np = points_b[:sample_n].cpu().numpy()
    ref_np = reflect_p[0, :sample_n].cpu().numpy()          # (S, M, 3)
    rot_np = rotate_p[0, :sample_n].cpu().numpy()

    fig, axes = plt.subplots(2, M, figsize=(M * 4, 8),
                              subplot_kw={'projection': '3d'})
    if M == 1:
        axes = axes.reshape(2, 1)

    for m in range(M):
        # 反射
        ax = axes[0, m]
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                   c='blue', s=2, alpha=0.5, label='原始')
        ax.scatter(ref_np[:, m, 0], ref_np[:, m, 1], ref_np[:, m, 2],
                   c='red', s=2, alpha=0.5, label='反射')
        ax.set_title(f"反射 {m}")
        ax.axis('equal')
        ax.legend(fontsize=6)

        # 旋转
        ax = axes[1, m]
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                   c='blue', s=2, alpha=0.5, label='原始')
        ax.scatter(rot_np[:, m, 0], rot_np[:, m, 1], rot_np[:, m, 2],
                   c='green', s=2, alpha=0.5, label='旋转')
        ax.set_title(f"旋转 {m}")
        ax.axis('equal')
        ax.legend(fontsize=6)

    fig.suptitle(f"推理结果可视化 — {result['model_id'][:16]} — loss: {result['loss']:.4f}",
                 fontsize=12)
    plt.tight_layout()
    plt.show()


# ===================== 主入口 =====================
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")

    # 加载模型
    model = PRSNet().to(DEVICE)          # 根据你的 PRSNet 定义调整参数
    model.load_state_dict(torch.load('prs-net.pth', map_location=DEVICE))
    print("模型已加载")

    # 加载数据集
    dataset = CombinedDataset(
        intermediate_data_dir='./shapenet_intermediate',
        device=DEVICE
    )

    # 损失函数
    loss_func = SymmetryLoss().to(DEVICE)

    # 推理
    results = evaluate(model, dataset, loss_func,
                       device=DEVICE, num_samples=5)

    # 可视化第一个样本
    visualize_symmetry(results[0], dataset)