import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
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


def visualize_symmetry_interactive(result, save_html=False):
    """
    交互式可视化：点云 + 反射平面 + 旋转轴
    使用 Plotly，支持三维旋转/缩放/平移
    """
    loss_func = SymmetryLoss()

    points_b   = result['points']            # (1, N, 3)
    planes     = result['planes']            # (1, 3, 4)
    quaternions = result['quaternions']      # (1, 3, 4)

    # —— 生成变换点云 ——
    with torch.no_grad():
        n, d = loss_func.convertToDirection(planes)
        reflect_p = loss_func.genReflectPoints(n, d, points_b)
        rotate_p  = loss_func.genRotationPoints(quaternions, points_b)

    M = planes.shape[1]
    N = points_b.shape[1]
    S = min(N, 1500)   # plotly 可以多画一些点

    pts_np = points_b[0, :S].cpu().numpy()
    ref_np = reflect_p[0, :S].cpu().numpy()   # (S, 3, 3)
    rot_np = rotate_p[0, :S].cpu().numpy()

    planes_np = planes[0].cpu().numpy()
    quats_np  = quaternions[0].cpu().numpy()

    axis_length = np.max(np.abs(pts_np)) * 1.1

    # ===== 6 个子图：上行反射，下行旋转 =====
    fig = make_subplots(
        rows=2, cols=M,
        specs=[[{'type': 'scene'}]*M, [{'type': 'scene'}]*M],
        subplot_titles=[
            *(f"Reflection {m}" for m in range(M)),
            *(f"Rotation {m}" for m in range(M)),
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )

    colors_original = 'blue'
    colors_ref = ['red', 'orange', 'darkred']
    colors_rot = ['green', 'lime', 'darkgreen']
    plane_colors = ['cyan', 'yellow', 'magenta']

    for m in range(M):
        row_ref, row_rot = 1, 2
        scene_ref = f'scene{m+1}'
        scene_rot = f'scene{M+m+1}'

        # ── 原始点云（两个子图都画） ──
        for row, scene in [(row_ref, scene_ref), (row_rot, scene_rot)]:
            fig.add_trace(go.Scatter3d(
                x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2],
                mode='markers',
                marker=dict(size=2, color=colors_original, opacity=0.4),
                name=f'Original',
                showlegend=(m == 0 and row == row_ref)
            ), row=row, col=m+1)

        # ── 反射点云 ──
        fig.add_trace(go.Scatter3d(
            x=ref_np[:, m, 0], y=ref_np[:, m, 1], z=ref_np[:, m, 2],
            mode='markers',
            marker=dict(size=2, color=colors_ref[m], opacity=0.6),
            name=f'Reflected {m}',
            showlegend=(m == 0)
        ), row=row_ref, col=m+1)

        # ── 反射平面 ──
        n_vec = planes_np[m, :3]
        d_val = planes_np[m, 3]
        n_vec = n_vec / (np.linalg.norm(n_vec) + 1e-8)
        P0 = -d_val * n_vec

        if abs(n_vec[0]) < 0.9:
            u = np.cross(n_vec, [1, 0, 0])
        else:
            u = np.cross(n_vec, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(n_vec, u)

        half = axis_length
        grid = np.array([
            P0 + s*u*half + t*v*half
            for s in [-1, 1] for t in [-1, 1]
        ])
        xx = grid[[0,2,1,3], 0].reshape(2,2)
        yy = grid[[0,2,1,3], 1].reshape(2,2)
        zz = grid[[0,2,1,3], 2].reshape(2,2)

        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, plane_colors[m]], [1, plane_colors[m]]],
            opacity=0.25, showscale=False,
            name=f'Plane {m}',
            showlegend=(m == 0)
        ), row=row_ref, col=m+1)

        # ── 旋转点云 ──
        fig.add_trace(go.Scatter3d(
            x=rot_np[:, m, 0], y=rot_np[:, m, 1], z=rot_np[:, m, 2],
            mode='markers',
            marker=dict(size=2, color=colors_rot[m], opacity=0.6),
            name=f'Rotated {m}',
            showlegend=(m == 0)
        ), row=row_rot, col=m+1)

        # ── 旋转轴（双向箭头） ──
        xyz = quats_np[m, 1:]
        axis_dir = xyz / (np.linalg.norm(xyz) + 1e-8)

        for sign, alpha, leg in [(1, 1.0, 'Axis'), (-1, 0.3, None)]:
            end = sign * axis_dir * axis_length
            # 用 Cone 代替箭头
            fig.add_trace(go.Cone(
                x=[end[0]*0.85], y=[end[1]*0.85], z=[end[2]*0.85],
                u=[axis_dir[0]*sign], v=[axis_dir[1]*sign], w=[axis_dir[2]*sign],
                sizemode='absolute', sizeref=axis_length*0.12,
                colorscale=[[0, 'magenta'], [1, 'magenta']],
                showscale=False,
                opacity=alpha,
                name='Rotation axis' if leg else None,
                showlegend=(m == 0 and leg is not None)
            ), row=row_rot, col=m+1)

        # ── 坐标轴设置 ──
        for scene_name in [scene_ref, scene_rot]:
            scene = fig.layout[scene_name]
            scene.xaxis.title = 'X'
            scene.yaxis.title = 'Y'
            scene.zaxis.title = 'Z'
            scene.xaxis.range = [-axis_length, axis_length]
            scene.yaxis.range = [-axis_length, axis_length]
            scene.zaxis.range = [-axis_length, axis_length]
            scene.aspectmode = 'cube'

    fig.update_layout(
        title=dict(
            text=f"PRS-Net — {result['model_id'][:16]} — Loss: {result['loss']:.4f}",
            font=dict(size=16)
        ),
        height=800,
        width=M * 450,
    )

    fig.show()

    fig.update_layout(
        title=dict(
            text=f"PRS-Net — {result['model_id'][:16]} — Loss: {result['loss']:.4f}",
            font=dict(size=16)
        ),
        height=800,
        width=M * 450,
    )

    fig.show()   # 浏览器打开

    if save_html:
        html_path = f"prsnet_{result['model_id'][:16]}.html"
        fig.write_html(html_path)
        print(f"✅ 已保存: {html_path}")

    return fig

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
        visualize_symmetry_interactive(results[m])
        print(results[m]["planes"])