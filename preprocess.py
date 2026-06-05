import os
import numpy as np
import torch
from scipy import ndimage
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import json
from typing import Tuple, List
import trimesh


class UnifiedShapeNetProcessor:
    """统一的 ShapeNet 处理器 - 同时处理体素和点云"""

    def __init__(self, shapenet_root: str, output_dir: str,
                 num_workers: int = None, num_samples: int = 5000):
        """
        Args:
            shapenet_root: ShapeNet 根目录
            output_dir: 输出文件夹（包含所有中间数据）
            num_workers: 并行进程数
            num_samples: 采样点数
        """
        self.shapenet_root = Path(shapenet_root)
        self.output_dir = Path(output_dir)

        # 创建子文件夹
        self.voxel_dir = self.output_dir / 'voxels'
        self.point_dir = self.output_dir / 'point_clouds'
        self.metadata_dir = self.output_dir / 'metadata'

        self.voxel_dir.mkdir(parents=True, exist_ok=True)
        self.point_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.num_workers = num_workers or os.cpu_count()
        self.num_samples = num_samples
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'voxel_success': 0,
            'point_success': 0
        }

    def find_all_models(self) -> list:
        """
        查找所有完整的模型

        Returns:
            list: [(category_id, model_id, obj_path, binvox_path), ...]
        """
        models = []

        for root, dirs, files in os.walk(self.shapenet_root):
            # 检查是否有 obj 和 binvox 文件
            has_obj = 'model_normalized.obj' in files
            has_binvox = 'model_normalized.solid.binvox' in files

            if has_obj and has_binvox:
                obj_path = os.path.join(root, 'model_normalized.obj')
                binvox_path = os.path.join(root, 'model_normalized.solid.binvox')

                parts = Path(obj_path).parts

                if len(parts) >= 4:
                    category_id = parts[-4]
                    model_id = parts[-3]
                    models.append((category_id, model_id, obj_path, binvox_path))

        return sorted(models)

    @staticmethod
    def _read_binvox(binvox_path: str) -> np.ndarray:
        """读取 binvox 文件"""
        with open(binvox_path, 'rb') as f:
            raw_data = f.read()

        header_text = raw_data[:512].decode('utf-8', errors='ignore')
        dims = [64, 64, 64]
        for line in header_text.split('\n'):
            if line.startswith('dim'):
                dims = list(map(int, line.split()[1:4]))
                break

        data_start = raw_data.find(b'\n\n') + 2
        rle_data = raw_data[data_start:]

        voxel_count = np.prod(dims)
        voxels = np.zeros(voxel_count, dtype=np.uint8)

        idx_pos = 0
        rle_pos = 0

        while rle_pos + 1 < len(rle_data) and idx_pos < voxel_count:
            value = rle_data[rle_pos]
            count = rle_data[rle_pos + 1]
            rle_pos += 2

            end_pos = min(idx_pos + count, voxel_count)
            voxels[idx_pos:end_pos] = value
            idx_pos = end_pos

        voxels = voxels.reshape(dims)
        return voxels

    @staticmethod
    def _resize_voxel_32(voxel_grid: np.ndarray) -> np.ndarray:
        """调整体素到 32x32x32"""
        if voxel_grid.shape == (32, 32, 32):
            return voxel_grid

        scale_factors = np.array([32.0, 32.0, 32.0]) / np.array(voxel_grid.shape)
        resized = ndimage.zoom(voxel_grid.astype(np.uint8), scale_factors, order=0)
        return (resized > 0).astype(np.uint8)

    @staticmethod
    def _sample_points_from_obj(obj_path: str, num_samples: int) -> np.ndarray:
        """从 OBJ 采样点"""
        mesh = trimesh.load(obj_path)

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [geom for geom in mesh.geometry.values()]
            )

        points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        return points.astype(np.float32)

    @staticmethod
    def process_single_model(args: Tuple) -> dict:
        """处理单个模型 - 同时生成体素和点云"""
        idx, category_id, model_id, obj_path, binvox_path, \
            voxel_dir, point_dir, metadata_dir, num_samples = args

        result = {
            'index': idx,
            'category': category_id,
            'model_id': model_id,
            'status': 'success',
            'voxel_file': None,
            'point_file': None,
            'metadata_file': None,
            'errors': []
        }

        try:
            # ========== 处理体素 ==========
            voxel_file = None
            voxel_occupancy = 0

            try:
                voxels = UnifiedShapeNetProcessor._read_binvox(binvox_path)
                voxels = UnifiedShapeNetProcessor._resize_voxel_32(voxels)
                voxel_occupancy = (voxels > 0).sum() / (32 ** 3) * 100

                # 只保存有效的体素（填充率 > 0.1%）
                if voxel_occupancy > 0.1:
                    voxel_filename = f'voxel_{idx:06d}_{category_id}_{model_id}.npy'
                    voxel_path = os.path.join(voxel_dir, voxel_filename)
                    np.save(voxel_path, voxels)
                    voxel_file = voxel_filename
                    result['voxel_file'] = voxel_filename
                    result['voxel_occupancy'] = float(voxel_occupancy)
                else:
                    result['errors'].append(f'体素填充率过低: {voxel_occupancy:.2f}%')

            except Exception as e:
                result['errors'].append(f'体素处理失败: {str(e)}')

            # ========== 处理点云 ==========
            point_file = None
            point_count = 0

            try:
                points = UnifiedShapeNetProcessor._sample_points_from_obj(obj_path, num_samples)
                point_count = len(points)

                point_filename = f'points_{idx:06d}_{category_id}_{model_id}.npy'
                point_path = os.path.join(point_dir, point_filename)
                np.save(point_path, points)
                point_file = point_filename
                result['point_file'] = point_filename
                result['point_count'] = int(point_count)

                # 记录点的范围
                result['point_range'] = {
                    'x': [float(points[:, 0].min()), float(points[:, 0].max())],
                    'y': [float(points[:, 1].min()), float(points[:, 1].max())],
                    'z': [float(points[:, 2].min()), float(points[:, 2].max())]
                }

            except Exception as e:
                result['errors'].append(f'点云处理失败: {str(e)}')

            # ========== 保存元数据 ==========
            if voxel_file and point_file:
                metadata = {
                    'index': idx,
                    'category': category_id,
                    'model_id': model_id,
                    'voxel_file': voxel_file,
                    'point_file': point_file,
                    'voxel_occupancy': float(voxel_occupancy),
                    'point_count': int(point_count),
                    'point_range': result['point_range']
                }

                metadata_filename = f'meta_{idx:06d}_{category_id}_{model_id}.json'
                metadata_path = os.path.join(metadata_dir, metadata_filename)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                result['metadata_file'] = metadata_filename

        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f'未知错误: {str(e)}')

        return result

    def process_all(self, save_manifest: bool = True) -> dict:
        """处理所有模型"""
        models = self.find_all_models()

        if not models:
            print('❌ 没有找到完整的模型文件！')
            print('   需要同时包含 model_normalized.obj 和 model_normalized.solid.binvox')
            return self.stats

        print(f'📦 找到 {len(models)} 个完整的模型')
        print(f'📂 输出目录: {self.output_dir}')
        print(f'   ├─ 体素: {self.voxel_dir}')
        print(f'   ├─ 点云: {self.point_dir}')
        print(f'   └─ 元数据: {self.metadata_dir}')
        print(f'🎯 采样点数: {self.num_samples}')
        print('-' * 80)

        # 准备任务
        tasks = [
            (idx, cat, mid, obj, binvox,
             str(self.voxel_dir), str(self.point_dir), str(self.metadata_dir),
             self.num_samples)
            for idx, (cat, mid, obj, binvox) in enumerate(models)
        ]

        # 多进程处理
        results = []
        with Pool(self.num_workers) as p:
            for result in tqdm(
                    p.imap_unordered(self.process_single_model, tasks),
                    total=len(tasks),
                    desc='处理进度',
                    unit='个'
            ):
                results.append(result)

        # 排序结果
        results_sorted = sorted(results, key=lambda x: x['index'])

        # 构建清单
        manifest = []
        for result in results_sorted:
            self.stats['total'] += 1

            if result['status'] == 'success':
                if result['voxel_file']:
                    self.stats['voxel_success'] += 1
                if result['point_file']:
                    self.stats['point_success'] += 1

                if result['voxel_file'] and result['point_file']:
                    self.stats['success'] += 1
                    manifest.append({
                        'index': result['index'],
                        'category': result['category'],
                        'model_id': result['model_id'],
                        'voxel_file': result['voxel_file'],
                        'point_file': result['point_file'],
                        'metadata_file': result['metadata_file'],
                        'voxel_occupancy': result.get('voxel_occupancy'),
                        'point_count': result.get('point_count'),
                        'point_range': result.get('point_range')
                    })

                    print(f"✅ [{result['index']:06d}] {result['category']}/{result['model_id']:16s} "
                          f"体素: {result['voxel_occupancy']:6.2f}% | "
                          f"点数: {result['point_count']:5d}")
                else:
                    if result['errors']:
                        error_msg = ' | '.join(result['errors'])
                        print(f"⚠️  [{result['index']:06d}] {result['category']}/{result['model_id']:16s} "
                              f"- {error_msg}")
                    self.stats['failed'] += 1
            else:
                self.stats['failed'] += 1
                error_msg = ' | '.join(result['errors']) if result['errors'] else '未知错误'
                print(f"❌ [{result['index']:06d}] {result['category']}/{result['model_id']:16s} "
                      f"- {error_msg}")

        # 保存总清单
        if save_manifest:
            manifest_path = self.output_dir / 'manifest.json'
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': self.stats['total'],
                    'success': self.stats['success'],
                    'failed': self.stats['failed'],
                    'voxel_success': self.stats['voxel_success'],
                    'point_success': self.stats['point_success'],
                    'voxel_size': 32,
                    'num_samples': self.num_samples,
                    'output_structure': {
                        'voxels': 'voxel_*.npy',
                        'point_clouds': 'points_*.npy',
                        'metadata': 'meta_*.json'
                    },
                    'files': manifest
                }, f, indent=2, ensure_ascii=False)
            print(f'\n📋 总清单已保存: {manifest_path}')

        # 统计
        print('\n' + '=' * 80)
        print(f'📊 总模型数: {self.stats["total"]}')
        print(f'✅ 完全成功（同时有体素和点云）: {self.stats["success"]}')
        print(f'✅ 体素生成: {self.stats["voxel_success"]}')
        print(f'✅ 点云生成: {self.stats["point_success"]}')
        print(f'❌ 失败: {self.stats["failed"]}')
        if self.stats['total'] > 0:
            success_rate = self.stats['success'] / self.stats['total'] * 100
            print(f'📊 成功率: {success_rate:.2f}%')
        print('=' * 80)

        return self.stats


class CombinedDataset(torch.utils.data.Dataset):
    """组合数据集 - 加载体素和点云对"""

    def __init__(self, intermediate_data_dir: str):
        """
        Args:
            intermediate_data_dir: 中间数据目录（由 UnifiedShapeNetProcessor 生成）
        """
        self.data_dir = Path(intermediate_data_dir)

        # 加载清单
        manifest_path = self.data_dir / 'manifest.json'
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.files = self.manifest['files']

        # 构建类别映射
        categories = set()
        for f in self.files:
            categories.add(f['category'])

        self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}

        print(f'✅ 加载了 {len(self.files)} 个模型对')
        print(f'📂 类别数: {len(self.category_to_idx)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        返回 (voxel, points, category_idx)

        Returns:
            voxel: (1, 32, 32, 32)
            points: (num_samples, 3)
            category_idx: int
        """
        file_info = self.files[idx]

        # 读取体素
        voxel_path = self.data_dir / 'voxels' / file_info['voxel_file']
        voxel = np.load(voxel_path)
        voxel_tensor = torch.from_numpy(voxel).float().unsqueeze(0)

        # 读取点云
        point_path = self.data_dir / 'point_clouds' / file_info['point_file']
        points = np.load(point_path)
        points_tensor = torch.from_numpy(points).float()

        # 获取类别
        category_idx = self.category_to_idx[file_info['category']]

        return voxel_tensor, points_tensor, category_idx


# ============= 使用示例 =============

if __name__ == '__main__':
    import time

    print("=" * 80)
    print("统一的 ShapeNet 数据处理")
    print("=" * 80)

    start_time = time.time()

    # 处理所有模型
    processor = UnifiedShapeNetProcessor(
        shapenet_root='ShapeNet',
        output_dir='./shapenet_intermediate',
        num_workers=None,  # 自动使用所有 CPU 核心
        num_samples=100
    )

    stats = processor.process_all(save_manifest=True)

    elapsed = time.time() - start_time
    print(f'\n⏱️  总耗时: {elapsed:.2f} 秒')

    # ============= 加载中间数据 =============

    print("\n" + "=" * 80)
    print("加载中间数据")
    print("=" * 80)

    dataset = CombinedDataset('./shapenet_intermediate')

    # 获取单个样本
    voxel, points, category = dataset[0]
    print(f'\n样本 0:')
    print(f'  体素形状: {voxel.shape}')
    print(f'  点云形状: {points.shape}')
    print(f'  类别索引: {category}')

    # ============= 使用 DataLoader =============

    print("\n" + "=" * 80)
    print("DataLoader 测试")
    print("=" * 80)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    for batch_voxels, batch_points, batch_categories in dataloader:
        print(f'\n批次:')
        print(f'  体素批次形状: {batch_voxels.shape}')
        print(f'  点云批次形状: {batch_points.shape}')
        print(f'  类别: {batch_categories}')
        break

    print('\n✅ 所有中间数据已准备好！')
    print('   你可以加载 CombinedDataset 来获取匹配的体素和点云对')