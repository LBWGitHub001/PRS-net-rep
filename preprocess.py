import os
import numpy as np
import torch
from scipy import ndimage
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import json
from typing import Tuple, List


class FastBinvoxReader:
    """高性能 Binvox 读取器"""

    @staticmethod
    def read_binvox_fast(filename: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        最高性能 binvox 读取（使用 NumPy 向量化）

        Args:
            filename: .binvox 文件路径

        Returns:
            voxel_grid: (D, H, W) uint8 数组
            dims: 原始维度
        """
        with open(filename, 'rb') as f:
            # 快速读取文件头（跳过文本解析，直接读取关键参数）
            header = f.read(1024).decode('utf-8', errors='ignore')

            # 快速提取维度
            dims = [64, 64, 64]  # 默认值
            if 'dim' in header:
                for line in header.split('\n'):
                    if line.startswith('dim'):
                        dims = list(map(int, line.split()[1:4]))
                        break

            # 跳到数据部分（定位到 '\n\n' 后）
            f.seek(0)
            content = f.read()
            data_start = content.find(b'\n\n') + 2
            f.seek(data_start)

            # 一次性读取所有数据到内存
            raw_data = f.read()

        # 使用 NumPy 快速解码 RLE
        voxels = np.zeros(np.prod(dims), dtype=np.uint8)

        idx = 0
        pos = 0

        while pos < len(raw_data) and idx < len(voxels):
            value = raw_data[pos]
            count = raw_data[pos + 1]
            pos += 2

            # 向量化填充
            end_idx = min(idx + count, len(voxels))
            voxels[idx:end_idx] = value
            idx = end_idx

        voxels = voxels.reshape(dims)
        return voxels, tuple(dims)


class HighPerformanceVoxelProcessor:
    """高性能体素处理器（优化版）"""

    def __init__(self, shapenet_root: str, output_dir: str, num_workers: int = None):
        """
        Args:
            shapenet_root: ShapeNet 根目录
            output_dir: 输出文件夹
            num_workers: 并行进程数（默认使用 CPU 核心数）
        """
        self.shapenet_root = Path(shapenet_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 自动检测 CPU 核心数
        if num_workers is None:
            num_workers = os.cpu_count()
        self.num_workers = num_workers

        self.stats = {'total': 0, 'success': 0, 'failed': 0}

    @staticmethod
    def fast_resize_32(voxel_grid: np.ndarray) -> np.ndarray:
        """
        高性能调整到 32x32x32（使用最近邻插值，最快）

        Args:
            voxel_grid: 原始体素网格

        Returns:
            resized: 32x32x32 体素
        """
        if voxel_grid.shape == (32, 32, 32):
            return voxel_grid

        # 计算缩放因子
        scale_factors = np.array([32.0, 32.0, 32.0]) / np.array(voxel_grid.shape)

        # 使用最近邻插值（order=0，最快）
        resized = ndimage.zoom(voxel_grid.astype(np.uint8), scale_factors, order=0)

        # 二值化
        return (resized > 127).astype(np.uint8)

    def find_binvox_files_fast(self, voxel_type: str = 'solid') -> List[Tuple]:
        """
        快速查找所有 binvox 文件（使用 os.walk）

        Args:
            voxel_type: 'solid' 或 'surface'

        Returns:
            list: [(category_id, model_id, file_path), ...]
        """
        binvox_files = []
        target_filename = f'model_normalized.{voxel_type}.binvox'

        for root, dirs, files in os.walk(self.shapenet_root):
            # 快速检查是否存在目标文件
            if target_filename in files:
                file_path = os.path.join(root, target_filename)

                # 从路径解析 category 和 model_id
                parts = Path(file_path).parts

                # 通常结构: ShapeNetCore.v2/CATEGORY_ID/MODEL_ID/models/...
                if len(parts) >= 4:
                    category_id = parts[-4]
                    model_id = parts[-3]
                    binvox_files.append((category_id, model_id, file_path))

        return sorted(binvox_files)

    @staticmethod
    def process_single_binvox_optimized(args: Tuple) -> dict:
        """
        优化版单文件处理（内联所有操作）

        Args:
            args: (idx, category_id, model_id, binvox_path, output_dir)

        Returns:
            result: 处理结果字典
        """
        idx, category_id, model_id, binvox_path, output_dir = args

        try:
            # 1. 快速读取 binvox
            with open(binvox_path, 'rb') as f:
                raw_data = f.read()

            # 解析头部获取维度
            header_text = raw_data[:512].decode('utf-8', errors='ignore')
            dims = [64, 64, 64]
            for line in header_text.split('\n'):
                if line.startswith('dim'):
                    dims = list(map(int, line.split()[1:4]))
                    break

            # 找到数据起始位置
            data_start = raw_data.find(b'\n\n') + 2
            rle_data = raw_data[data_start:]

            # 2. 快速 RLE 解码（向量化）
            voxel_count = np.prod(dims)
            voxels = np.zeros(voxel_count, dtype=np.uint8)

            idx_pos = 0
            rle_pos = 0

            while rle_pos < len(rle_data) - 1 and idx_pos < voxel_count:
                value = rle_data[rle_pos]
                count = rle_data[rle_pos + 1]
                rle_pos += 2

                end_pos = min(idx_pos + count, voxel_count)
                voxels[idx_pos:end_pos] = value
                idx_pos = end_pos

            voxels = voxels.reshape(dims)

            # 3. 快速调整到 32x32x32
            if voxels.shape != (32, 32, 32):
                scale = np.array([32.0, 32.0, 32.0]) / np.array(voxels.shape)
                voxels = ndimage.zoom(voxels, scale, order=0)
                voxels = (voxels > 127).astype(np.uint8)

            # 4. 快速保存为 .npy
            output_filename = f'voxel_{idx:06d}_{category_id}_{model_id}.npy'
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, voxels)

            # 计算填充率
            occupancy = (voxels > 0).sum() / (32 ** 3) * 100

            return {
                'status': 'success',
                'index': idx,
                'category': category_id,
                'model_id': model_id,
                'output': output_filename,
                'occupancy': float(occupancy)
            }

        except Exception as e:
            return {
                'status': 'failed',
                'index': idx,
                'category': category_id,
                'model_id': model_id,
                'error': str(e)
            }

    def process_all_fast(self, voxel_type: str = 'solid', save_manifest: bool = True) -> dict:
        """
        最高性能批量处理

        Args:
            voxel_type: 'solid' 或 'surface'
            save_manifest: 是否保存清单

        Returns:
            stats: 统计信息
        """
        # 查找所有 binvox 文件
        binvox_files = self.find_binvox_files_fast(voxel_type)

        if not binvox_files:
            print(f'❌ 没有找到 {voxel_type} 类型的 binvox 文件！')
            return self.stats

        print(f'📦 找到 {len(binvox_files)} 个 binvox 文件')
        print(f'📂 输出目录: {self.output_dir}')
        print(f'⚙️  CPU 核心数: {self.num_workers}')
        print(f'🚀 模式: 最高性能 (向量化 + 多进程)')
        print('-' * 80)

        # 准备任务列表
        tasks = [
            (idx, cat, mid, fp, str(self.output_dir))
            for idx, (cat, mid, fp) in enumerate(binvox_files)
        ]

        # 多进程处理
        results = []
        with Pool(self.num_workers) as p:
            for result in tqdm(
                    p.imap_unordered(self.process_single_binvox_optimized, tasks),
                    total=len(tasks),
                    desc='处理进度',
                    unit='个',
                    unit_scale=True
            ):
                results.append(result)

        # 排序结果
        results_sorted = sorted(
            [r for r in results if 'index' in r],
            key=lambda x: x['index']
        )

        # 构建清单
        manifest = []
        for result in results_sorted:
            if result['status'] == 'success':
                self.stats['success'] += 1
                manifest.append(result)
                print(f"✅ [{result['index']:06d}] {result['category']}/{result['model_id']:16s} "
                      f"→ {result['output']} (填充率: {result['occupancy']:6.2f}%)")
            else:
                self.stats['failed'] += 1
                print(f"❌ [{result['index']:06d}] {result['category']}/{result['model_id']:16s} - {result['error']}")

        self.stats['total'] = len(binvox_files)

        # 保存清单
        if save_manifest:
            manifest_path = self.output_dir / 'manifest.json'
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': self.stats['total'],
                    'success': self.stats['success'],
                    'failed': self.stats['failed'],
                    'voxel_size': 32,
                    'voxel_type': voxel_type,
                    'cpu_cores': self.num_workers,
                    'files': manifest
                }, f, indent=2, ensure_ascii=False)
            print(f'\n📋 清单已保存: {manifest_path}')

        # 统计信息
        print('\n' + '=' * 80)
        print(f'✅ 成功: {self.stats["success"]}')
        print(f'❌ 失败: {self.stats["failed"]}')
        if self.stats['total'] > 0:
            success_rate = self.stats['success'] / self.stats['total'] * 100
            print(f'📊 成功率: {success_rate:.2f}%')
        print('=' * 80)

        return self.stats


# ============= 极速版本（一行代码） =============

def process_shapenet_binvox_ultra_fast(
        shapenet_root: str,
        output_dir: str,
        voxel_type: str = 'solid'
) -> dict:
    """
    超高性能一行处理

    Args:
        shapenet_root: ShapeNet 根目录
        output_dir: 输出目录
        voxel_type: 'solid' 或 'surface'

    Returns:
        stats: 统计信息
    """
    processor = HighPerformanceVoxelProcessor(shapenet_root, output_dir)
    return processor.process_all_fast(voxel_type=voxel_type, save_manifest=True)


# ============= 使用示例 =============

if __name__ == '__main__':
    import time

    # 记录开始时间
    start_time = time.time()

    # 1. 创建高性能处理器
    processor = HighPerformanceVoxelProcessor(
        shapenet_root='ShapeNet',
        output_dir='./voxels',
        num_workers=None  # 自动使用所有 CPU 核心
    )

    # 2. 处理所有 binvox 文件
    stats = processor.process_all_fast(voxel_type='solid', save_manifest=True)

    # 计算耗时
    elapsed = time.time() - start_time
    print(f'\n⏱️  总耗时: {elapsed:.2f} 秒')
    print(f'📊 平均速度: {stats["total"] / elapsed:.1f} 文件/秒')