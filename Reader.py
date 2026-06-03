import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Tuple, Optional, List
import os


class VoxelDataset(Dataset):
    """体素数据集（从 .npy 文件加载）"""

    def __init__(
            self,
            voxel_dir: str,
            manifest_path: Optional[str] = None,
            transform: Optional[callable] = None,
            device: str = 'cpu'
    ):
        """
        Args:
            voxel_dir: 体素文件所在目录
            manifest_path: manifest.json 路径（可选，用于获取元数据）
            transform: 数据变换函数（可选）
            device: 'cpu' 或 'cuda'
        """
        self.voxel_dir = Path(voxel_dir)
        self.device = device
        self.transform = transform

        # 查找所有 .npy 文件
        self.voxel_files = sorted(self.voxel_dir.glob('voxel_*.npy'))

        if len(self.voxel_files) == 0:
            raise ValueError(f'在 {voxel_dir} 中没有找到 .npy 文件！')

        # 加载清单（如果存在）
        self.manifest = None
        self.category_to_idx = {}

        if manifest_path and Path(manifest_path).exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)

            # 构建类别到索引的映射
            categories = set()
            for file_info in self.manifest.get('files', []):
                categories.add(file_info['category'])

            self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}

        print(f'✅ 加载了 {len(self.voxel_files)} 个体素文件')
        if self.category_to_idx:
            print(f'📂 共有 {len(self.category_to_idx)} 个类别')

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.voxel_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int], Optional[str]]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (voxel_tensor, category_idx, metadata)
        """
        # 1. 加载体素数据
        voxel_path = self.voxel_files[idx]
        voxel_data = np.load(voxel_path)  # (32, 32, 32)

        # 2. 转换为 PyTorch 张量
        voxel_tensor = torch.from_numpy(voxel_data).float()
        voxel_tensor = voxel_tensor.unsqueeze(0)  # (1, 32, 32, 32) - 添加通道维度

        # 3. 获取类别标签（如果有清单）
        category_idx = None
        category_label = None

        if self.manifest:
            # 从文件名解析类别
            filename = voxel_path.stem  # voxel_000000_02691156_1a04e3ea
            parts = filename.split('_')
            if len(parts) >= 3:
                category_label = parts[2]
                category_idx = self.category_to_idx.get(category_label, -1)

        # 4. 应用变换（如果有）
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)

        # 5. 移动到指定设备
        voxel_tensor = voxel_tensor.to(self.device)

        return voxel_tensor, category_idx, category_label

    def get_category_info(self) -> dict:
        """获取类别信息"""
        if not self.category_to_idx:
            return {}

        return {
            'num_classes': len(self.category_to_idx),
            'category_to_idx': self.category_to_idx,
            'idx_to_category': {v: k for k, v in self.category_to_idx.items()}
        }


class VoxelDataLoader:
    """体素数据加载器（便利类）"""

    def __init__(
            self,
            voxel_dir: str,
            manifest_path: Optional[str] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            shuffle: bool = True,
            pin_memory: bool = True
    ):
        """
        Args:
            voxel_dir: 体素文件目录
            manifest_path: manifest.json 路径
            batch_size: 批次大小
            num_workers: 加载数据的工作进程数
            shuffle: 是否打乱数据
            pin_memory: 是否锁定内存（GPU 训练时推荐）
        """
        self.dataset = VoxelDataset(
            voxel_dir=voxel_dir,
            manifest_path=manifest_path,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_category_info(self):
        return self.dataset.get_category_info()


# ============= 使用示例 =============

if __name__ == '__main__':
    # 1. 基本使用
    print("=" * 80)
    print("1. 基本使用")
    print("=" * 80)

    dataset = VoxelDataset(
        voxel_dir='./voxels',
        manifest_path='./voxels/manifest.json',
        device='cpu'
    )

    # 获取单个样本
    voxel, category_idx, category_label = dataset[0]
    print(f'单个样本:')
    print(f'  体素张量形状: {voxel.shape}')  # (1, 32, 32, 32)
    print(f'  类别索引: {category_idx}')
    print(f'  类别标签: {category_label}')
    print(f'  数据类型: {voxel.dtype}')
    print()

    # 2. 使用 DataLoader（推荐）
    print("=" * 80)
    print("2. 使用 DataLoader")
    print("=" * 80)

    dataloader = VoxelDataLoader(
        voxel_dir='./voxels',
        manifest_path='./voxels/manifest.json',
        batch_size=32,
        num_workers=4,
        shuffle=True
    )

    # 获取类别信息
    category_info = dataloader.get_category_info()
    print(f'类别信息:')
    print(f'  总类别数: {category_info["num_classes"]}')
    print(f'  类别映射: {list(category_info["category_to_idx"].items())[:5]}...')
    print()

    # 迭代加载数据
    print("迭代数据:")
    for batch_idx, (voxels, categories, labels) in enumerate(dataloader):
        print(f'  批次 {batch_idx}:')
        print(f'    体素张量: {voxels.shape}')  # (32, 1, 32, 32, 32)
        print(f'    类别索引: {categories}')
        print(f'    类别标签: {labels}')

        if batch_idx == 2:  # 仅显示前 3 个批次
            break
    print()

    # 4. 创建训练和验证集
    print("=" * 80)
    print("4. 划分训练/验证集")
    print("=" * 80)

    from torch.utils.data import random_split

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print()

    # 5. 检查数据
    print("=" * 80)
    print("5. 数据检查")
    print("=" * 80)

    batch_voxels, batch_categories, batch_labels = next(iter(train_loader))
    print(f'批次统计:')
    print(f'  形状: {batch_voxels.shape}')
    print(f'  数据类型: {batch_voxels.dtype}')
    print(f'  值范围: [{batch_voxels.min():.2f}, {batch_voxels.max():.2f}]')
    print(f'  非零体素: {(batch_voxels > 0).sum().item()} / {batch_voxels.numel()}')