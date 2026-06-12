import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Tuple, Optional
from scipy.spatial import KDTree
import os


class CombinedDataset(Dataset):
    """组合数据集 - 加载体素和点云对"""

    def __init__(
            self,
            intermediate_data_dir: str,
            transform: Optional[callable] = None,
            device: str = 'cpu'
    ):
        """
        Args:
            intermediate_data_dir: 中间数据目录（由 UnifiedShapeNetProcessor 生成）
            transform: 数据变换函数（可选）
            device: 'cpu' 或 'cuda'
        """
        self.data_dir = Path(intermediate_data_dir)
        self.device = device
        self.transform = transform
        self.nearest_idx_maps = []
        self.nearest_pts_maps = []

        # 加载清单
        manifest_path = self.data_dir / 'manifest.json'
        if not manifest_path.exists():
            raise ValueError(f'在 {intermediate_data_dir} 中没有找到 manifest.json！')

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.files = self.manifest['files']

        if len(self.files) == 0:
            raise ValueError(f'清单中没有有效的文件！')

        # 预加载所有数据
        self._preload_data()

        print(f'✅ 加载了 {len(self.files)} 个模型对')

    def _preload_data(self):
        """预加载所有体素和点云数据"""
        print("🔄 预加载数据...")

        self.voxel_tensors = []
        self.point_tensors = []
        self.bbox_mins = []
        self.bbox_maxs = []

        voxel_dir = self.data_dir / 'voxels'
        point_dir = self.data_dir / 'point_clouds'

        for file_info in self.files:
            try:
                # 加载体素
                voxel_path = voxel_dir / file_info['voxel_file']
                voxel_data = np.load(voxel_path)
                voxel_tensor = torch.from_numpy(voxel_data).float()
                voxel_tensor = voxel_tensor.unsqueeze(0)  # (1, 32, 32, 32)
                bboox_min, bboox_max = 0, 0

                # 加载点云
                point_path = point_dir / file_info['point_file']
                points_data = np.load(point_path)
                points_tensor = torch.from_numpy(points_data).float()

                # 应用变换
                if self.transform:
                    voxel_tensor = self.transform(voxel_tensor)

                self.voxel_tensors.append(voxel_tensor)
                self.point_tensors.append(points_tensor)

                meta_path = self.data_dir / 'metadata' / file_info.get('metadata_file', '')
                if meta_path.exists():
                    with open(meta_path, 'r') as mf:
                        meta = json.load(mf)
                    bbox_min = np.array(meta['min'], dtype=np.float32)
                    bbox_max = np.array(meta['max'], dtype=np.float32)
                    self.bbox_mins.append(bbox_min)
                    self.bbox_maxs.append(bbox_max)
                else:
                    # 回退：用点云范围估算
                    bbox_min = points_data.min(axis=0)
                    bbox_max = points_data.max(axis=0)
                    self.bbox_mins.append(bbox_min.astype(np.float32))
                    self.bbox_maxs.append(bbox_max.astype(np.float32))

                # ---- 预计算：体素→最近点 ----
                near_idx, near_pts = self._compute_voxel_nearest(
                    voxel_data, points_data, bbox_min, bbox_max
                )
                near_idx_t = torch.from_numpy(near_idx)  # (32,32,32) int32
                near_pts_t = torch.from_numpy(near_pts)  # (32,32,32,3) float32
                self.nearest_idx_maps.append(near_idx_t)
                self.nearest_pts_maps.append(near_pts_t)


            except Exception as e:
                print(f"⚠️  跳过文件 {file_info['voxel_file']}: {str(e)}")
                continue

        # 栈合并
        self.voxel_tensors = torch.stack(self.voxel_tensors)  # (N, 1, 32, 32, 32)

        # 移到设备
        self.voxel_tensors = self.voxel_tensors.to(self.device)

        print(f"✅ 预加载完成！")
        print(f"   体素张量形状: {self.voxel_tensors.shape}")
        print(f"   内存占用: {self.voxel_tensors.element_size() * self.voxel_tensors.nelement() / 1024**2:.2f} MB")

    @staticmethod
    def _compute_voxel_nearest(voxel_np, pts_np, bbox_min, bbox_max):
        """
        voxel_np : (32, 32, 32) uint8
        pts_np   : (N, 3) float32
        bbox_min : (3,) float32  — metadata 精确包围盒
        bbox_max : (3,) float32
        返回:
          nearest_idx : (32, 32, 32) int32  空体素=-1
          nearest_pts : (32, 32, 32, 3) float32  空体素=0
        """
        V = 32
        size = bbox_max - bbox_min  # (3,)

        # 体素中心坐标（包围盒空间）
        coords = (np.arange(V) + 0.5) / V  # (32,)
        ii, jj, kk = np.meshgrid(coords, coords, coords, indexing='ij')
        cx = bbox_min[0] + ii * size[0]
        cy = bbox_min[1] + jj * size[1]
        cz = bbox_min[2] + kk * size[2]
        centers = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)  # (32768, 3)

        # 空体素掩码
        empty_mask = (voxel_np == 0).reshape(-1)

        # KDTree 查询（只查非空体素以加速）
        tree = KDTree(pts_np)
        dist, idx = tree.query(centers, workers=1)

        # 重塑
        idx_3d = idx.astype(np.int32).reshape(V, V, V)
        idx_3d[empty_mask.reshape(V, V, V)] = -1

        # 最近点坐标 (32,32,32,3)，空体素填 0
        pts_3d = pts_np[idx].reshape(V, V, V, 3).astype(np.float32)
        pts_3d[empty_mask.reshape(V, V, V)] = 0.0

        return idx_3d, pts_3d

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,tuple[torch.Tensor, torch.Tensor]]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (voxel_tensor, points_tensor, category_idx)
        """
        voxel = self.voxel_tensors[idx]  # (1, 32, 32, 32)
        points = self.point_tensors[idx]  # (num_samples, 3)

        return voxel, points, (self.nearest_idx_maps[idx], self.nearest_pts_maps[idx])

    def get_model_info(self, idx: int) -> dict:
        """获取模型的完整信息"""
        return self.files[idx]


class CombinedDataLoader:
    """组合数据加载器"""

    def __init__(
            self,
            intermediate_data_dir: str,
            batch_size: int = 32,
            num_workers: int = 0,
            shuffle: bool = True,
            pin_memory: bool = True,
            device: str = 'cpu',
            reader: str = 'tensor'
    ):
        """
        Args:
            intermediate_data_dir: 中间数据目录
            batch_size: 批次大小
            num_workers: 加载数据的工作进程数（预加载时应为 0）
            shuffle: 是否打乱数据
            pin_memory: 是否锁定内存
            device: 'cpu' 或 'cuda'
        """
        self.dataset = CombinedDataset(
            intermediate_data_dir=intermediate_data_dir,
            device=device
        )
        if reader == 'list':
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and device == 'cuda',
                collate_fn=self.collate_points_
            )
        elif reader == 'tensor':
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and device == 'cuda'
            )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    @staticmethod
    def collate_points_(batch):
        voxels = torch.stack([item[0] for item in batch])  # (B, 1, 32, 32, 32)
        points_list = [item[1] for item in batch]  # 保持为 list，不 stack
        return voxels, points_list



# ============= 使用示例 =============

if __name__ == '__main__':
    # 1. 基本使用
    print("=" * 80)
    print("1. 基本使用")
    print("=" * 80)

    dataset = CombinedDataset(
        intermediate_data_dir='./shapenet_intermediate',
        device='cpu'
    )

    # 获取单个样本
    voxel, points = dataset[0]
    print(f'\n单个样本:')
    print(f'  体素张量形状: {voxel.shape}')  # (1, 32, 32, 32)
    print(f'  点云张量形状: {points.shape}')  # (num_samples, 3)
    print(f'  体素数据类型: {voxel.dtype}')
    print(f'  点云数据类型: {points.dtype}')

    # 获取模型信息
    model_info = dataset.get_model_info(0)
    print(f'\n模型信息:')
    print(f'  模型ID: {model_info["model_id"]}')
    print(f'  类别: {model_info["category"]}')
    print(f'  体素占有率: {model_info["voxel_occupancy"]:.2f}%')
    print(f'  点云数量: {model_info["point_count"]}')
    print()

    # 2. 使用 DataLoader（推荐）
    print("=" * 80)
    print("2. 使用 DataLoader")
    print("=" * 80)

    dataloader = CombinedDataLoader(
        intermediate_data_dir='./shapenet_intermediate',
        batch_size=32,
        num_workers=0,  # 预加载时设置为 0
        shuffle=True,
        device='cpu'
    )

    # 迭代加载数据
    print("迭代数据:")
    for batch_idx, (batch_voxels, batch_points) in enumerate(dataloader):
        print(f'\n  批次 {batch_idx}:')
        print(f'    体素批次: {batch_voxels.shape}')  # (32, 1, 32, 32, 32)
        print(f'    点云数量: {len(batch_points)}')  # (32, num_samples, 3)

        if batch_idx == 2:  # 仅显示前 3 个批次
            break
    print()

    # 3. 创建训练和验证集
    print("=" * 80)
    print("3. 划分训练/验证集")
    print("=" * 80)

    from torch.utils.data import random_split

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=CombinedDataLoader.collate_points_
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=CombinedDataLoader.collate_points_
    )

    print(f'\n训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print()

    # 4. 检查数据
    print("=" * 80)
    print("4. 数据检查")
    print("=" * 80)

    batch_voxels, batch_points = next(iter(train_loader))

    print(f'\n批次统计:')
    print(f'  体素形状: {batch_voxels.shape}')
    print(f'  体素数据类型: {batch_voxels.dtype}')
    print(f'  体素值范围: [{batch_voxels.min():.2f}, {batch_voxels.max():.2f}]')
    print(f'  体素非零: {(batch_voxels > 0).sum().item()} / {batch_voxels.numel()}')

    print(f'\n  点云类型: list of tensors（变长）')
    print(f'  批次样本数: {len(batch_points)}')
    print(f'  各样本点数: {[p.shape[0] for p in batch_points]}')

    all_points = torch.cat(batch_points, dim=0)
    print(f'  点云坐标范围（跨所有样本）:')
    print(f'    X: [{all_points[:, 0].min():.4f}, {all_points[:, 0].max():.4f}]')
    print(f'    Y: [{all_points[:, 1].min():.4f}, {all_points[:, 1].max():.4f}]')
    print(f'    Z: [{all_points[:, 2].min():.4f}, {all_points[:, 2].max():.4f}]')

    # 5. GPU 支持
    print("\n" + "=" * 80)
    print("5. GPU 支持示例")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f'\n✅ 检测到 GPU: {torch.cuda.get_device_name(0)}')

        gpu_dataset = CombinedDataset(
            intermediate_data_dir='./shapenet_intermediate',
            device='cuda'
        )

        gpu_loader = CombinedDataLoader(
            intermediate_data_dir='./shapenet_intermediate',
            batch_size=64,
            device='cuda'
        )

        voxel, points, category = gpu_dataset[0]
        print(f'  体素设备: {voxel.device}')
        print(f'  点云设备: {points.device}')

        batch_voxels, batch_points, _ = next(iter(gpu_loader))
        print(f'  批次体素设备: {batch_voxels.device}')
        print(f'  批次点云设备: {batch_points.device}')
    else:
        print('\n⚠️  未检测到 GPU，使用 CPU')