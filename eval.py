import torch
import numpy as np
import matplotlib.pyplot as plt
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
        for i in range(min(num_samples, len(dataset))):
            # ✅ __getitem__ 返回 5 个值
            voxel, points, idx_map, pts_map, (bbox_max, bbox_min) = dataset[i]

            # 拼 batch 维度 (1, ...)
            voxel_b   = voxel.unsqueeze(0).to(device)        # (1, 1, 32, 32, 32)
            points_b  = points.unsqueeze(0).to(device)       # (1, N, 3)
            idx_map_b = idx_map.unsqueeze(0).to(device)      # (1, 32, 32, 32)
            pts_map_b = pts_map.unsqueeze(0).to(device)      # (1, 32, 32, 32, 3)
            max_min = (bbox_max.unsqueeze(0), bbox_min.unsqueeze(0))

            # 推理
            planes, quaternions = model(voxel_b)             # (1, 3, 4), (1, 3, 4)

            # ✅ 用 Tensor 模式算 loss（不是 list）
            loss = loss_func(planes, quaternions,
                             points_b,                       # Tensor (1, N, 3)
                             idx_map_b, pts_map_b, max_min)

            loss_val = loss.item()
            total_loss += loss_val

            info = dataset.get_model_info(i)
            results.append({
                'idx': i,
                'model_id': info['model_id'],
                'category': info['category'],
                'loss': loss_val,
                'voxel': voxel_b,
                'points': points_b,
                'planes': planes,
                'quaternions': quaternions,
                # ✅ 可视化需要用到的数据也存起来
                'pts_map': pts_map,
                'bbox_max': bbox_max,
                'bbox_min': bbox_min,
            })

            print(f"[eval {i+1}/{num_samples}] "
                  f"{info['model_id'][:16]}  "
                  f"loss: {loss_val:.6f}")

    avg_loss = total_loss / num_samples
    print(f"\n{'='*50}")
    print(f"平均 Loss: {avg_loss:.6f}")
    print(f"{'='*50}")

    return results


def draw_plane(ax, n, d, points_np, color='cyan', alpha=0.25):
    """
    在 ax 上画反射平面 n·P + d = 0
    n: (3,) 单位法向量
    d: 标量偏移
    points_np: (N,3) 点云，用于自动确定平面大小
    """
    n = n / (np.linalg.norm(n) + 1e-8)
    P0 = -d * n                               # 平面上离原点最近的点

    # 两个正交切向量
    if abs(n[0]) < 0.9:
        u = np.cross(n, [1, 0, 0])
    else:
        u = np.cross(n, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    # 根据点云范围自动设定平面大小
    extent = np.max(np.abs(points_np)) * 1.1
    half = extent

    corners = np.array([
        P0 + half * (-u - v),
        P0 + half * (-u + v),
        P0 + half * ( u + v),
        P0 + half * ( u - v),
    ])
    xx = corners[[0, 1, 3, 2], 0].reshape(2, 2)
    yy = corners[[0, 1, 3, 2], 1].reshape(2, 2)
    zz = corners[[0, 1, 3, 2], 2].reshape(2, 2)

    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, edgecolor='none')


def draw_rotation_axis(ax, quat, length, color='magenta', lw=2):
    """
    画旋转轴（双向箭头）
    quat: (4,) 四元数 (w, x, y, z)
    """
    xyz = quat[1:]                              # (x, y, z)
    axis = xyz / (np.linalg.norm(xyz) + 1e-8)   # 单位方向

    ax.quiver(0, 0, 0,
              axis[0] * length, axis[1] * length, axis[2] * length,
              color=color, linewidth=lw, arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0,
              -axis[0] * length, -axis[1] * length, -axis[2] * length,
              color=color, linewidth=lw, arrow_length_ratio=0.15, alpha=0.3)


def visualize_symmetry(result):
    """
    可视化：原始点云 + 反射点云 + 旋转点云 + 反射平面 + 旋转轴
    """
    loss_func = SymmetryLoss()

    points_b   = result['points']        # (1, N, 3)
    planes     = result['planes']        # (1, 3, 4)
    quaternions = result['quaternions']  # (1, 3, 4)

    # —— 生成变换点云 ——
    with torch.no_grad():
        n, d = loss_func.convertToDirection(planes)
        reflect_p = loss_func.genReflectPoints(n, d, points_b)
        rotate_p  = loss_func.genRotationPoints(quaternions, points_b)

    M = planes.shape[1]
    N = points_b.shape[1]
    S = min(N, 500)

    pts_np = points_b[0, :S].cpu().numpy()        # (S, 3)
    ref_np = reflect_p[0, :S].cpu().numpy()       # (S, 3, 3)
    rot_np = rotate_p[0, :S].cpu().numpy()        # (S, 3, 3)

    # 平面参数 & 四元数 → numpy
    planes_np     = planes[0].cpu().numpy()        # (3, 4)
    quats_np      = quaternions[0].cpu().numpy()   # (3, 4)

    axis_length = np.max(np.abs(pts_np)) * 1.1

    # —— 画图 ——
    fig, axes = plt.subplots(2, M, figsize=(M * 4, 8),
                              subplot_kw={'projection': '3d'})
    if M == 1:
        axes = np.array(axes).reshape(2, 1)

    for m in range(M):
        # ===== 反射 =====
        ax = axes[0, m]
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                   c='blue', s=2, alpha=0.5, label='Original')
        ax.scatter(ref_np[:, m, 0], ref_np[:, m, 1], ref_np[:, m, 2],
                   c='red', s=2, alpha=0.5, label='Reflected')

        # 画反射平面
        n_vec = planes_np[m, :3]
        d_val = planes_np[m, 3]
        n_vec = n_vec / (np.linalg.norm(n_vec) + 1e-8)      # 确保单位
        draw_plane(ax, n_vec, d_val, pts_np, color='cyan', alpha=0.25)

        ax.set_title(f"Reflection {m}")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(fontsize=6)

        # ===== 旋转 =====
        ax = axes[1, m]
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                   c='blue', s=2, alpha=0.5, label='Original')
        ax.scatter(rot_np[:, m, 0], rot_np[:, m, 1], rot_np[:, m, 2],
                   c='green', s=2, alpha=0.5, label='Rotated')

        # 画旋转轴
        draw_rotation_axis(ax, quats_np[m], axis_length, color='magenta', lw=2)

        ax.set_title(f"Rotation {m}")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(fontsize=6)

    fig.suptitle(f"PRS-Net Inference — {result['model_id'][:16]} — Loss: {result['loss']:.4f}",
                 fontsize=12)
    plt.tight_layout()
    plt.show()


# ===================== 主入口 =====================
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")

    # 加载模型
    model = PRSNet().to(DEVICE)
    checkpoint = torch.load('prs-net-best.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
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

    # 可视化第一个
    for m in range(5):
        visualize_symmetry(results[m])