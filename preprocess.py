import os
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import json
from typing import Tuple
import trimesh
from scipy.spatial import KDTree


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
        self.map_dir = self.output_dir / 'maps'
        self.metadata_dir = self.output_dir / 'metadata'

        self.voxel_dir.mkdir(parents=True, exist_ok=True)
        self.point_dir.mkdir(parents=True, exist_ok=True)
        self.map_dir.mkdir(parents=True, exist_ok=True)
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
        models = []
        for root, dirs, files in os.walk(self.shapenet_root):
            has_obj = 'model_normalized.obj' in files
            has_binvox = 'model_normalized.solid.binvox' in files
            has_json = 'model_normalized.json' in files  # <-- 加这行

            if has_obj and has_binvox and has_json:  # <-- 必须3个都有
                obj_path = os.path.join(root, 'model_normalized.obj')
                binvox_path = os.path.join(root, 'model_normalized.solid.binvox')
                json_path = os.path.join(root, 'model_normalized.json')  # <-- 加这行

                parts = Path(obj_path).parts
                if len(parts) >= 4:
                    category_id = parts[-4]
                    model_id = parts[-3]
                    models.append((category_id, model_id, obj_path, binvox_path, json_path))  # <-- 加json
        return sorted(models)

    # ----------------------
    # 【正确】读 binvox 格式（替换你错误的手动解析）
    # ----------------------
    @staticmethod
    def _read_binvox(binvox_path):
        import binvox_rw  # 必须用官方库
        with open(binvox_path, 'rb') as f:
            voxel = binvox_rw.read_as_3d_array(f)
        return voxel.data  # (D, W, H) 正确3D体素

    # ----------------------
    # 【正确】缩放到32^3（不会变平面）
    # ----------------------
    @staticmethod
    def _resize_voxel_32(voxel, target_size=32):
        import skimage.transform
        from scipy import ndimage

        # ✅ 方案 A：使用双线性插值 + 更低阈值（推荐）
        voxel = skimage.transform.resize(
            voxel.astype(np.float32),
            (target_size, target_size, target_size),
            order=1,  # ← 双线性插值，保留细结构
            preserve_range=True,
            anti_aliasing=True  # ← 抗锯齿，更平滑
        )
        return (voxel > 0.3).astype(np.uint8)  # ← 降低阈值

    @staticmethod
    def _sample_points_from_obj(obj_path: str, num_samples: int) -> np.ndarray:
        """从 OBJ 采样点"""
        mesh = trimesh.load(obj_path)

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [geom for geom in mesh.geometry.values()]
            )

        points, _ = trimesh.sample.sample_surface(mesh, num_samples)

        centroid = points.mean(axis=0)
        points = points - centroid
        return points.astype(np.float32)

    @staticmethod
    def _compute_voxel_nearest(voxel_np, pts_np, bbox_min, bbox_max, V=32):
        """
        voxel_np : (32, 32, 32) uint8
        pts_np   : (N, 3) float32
        bbox_min : (3,) float32  — metadata 精确包围盒
        bbox_max : (3,) float32
        返回:
          nearest_idx : (32, 32, 32) int32  空体素=-1
          nearest_pts : (32, 32, 32, 3) float32  空体素=0
        """
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

        # 最近点坐标 (32,32,32,3)，空体素填 nan
        pts_3d = pts_np[idx].reshape(V, V, V, 3).astype(np.float32)
        # pts_3d[empty_mask.reshape(V, V, V)] = -1.0 #TODO 到0.0的位置值应该比较大，这里可能是一个隐患

        return idx_3d, pts_3d

    @staticmethod
    def process_single_model(args: Tuple) -> dict:
        idx, category_id, model_id, obj_path, binvox_path, json_path, \
            voxel_dir, point_dir, map_dir, metadata_dir, num_samples = args

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
            # ========== 【核心】读取官方 json 质心 + 包围盒 ==========
            with open(json_path, 'r', encoding='utf-8') as f:
                shape_info = json.load(f)
            centroid_gt = np.array(shape_info['centroid'], dtype=np.float32)
            min_gt = np.array(shape_info['min'], dtype=np.float32)
            max_gt = np.array(shape_info['max'], dtype=np.float32)

            # ========== 处理点云：使用官方质心居中 ==========
            point_file = None
            full_point_file = None
            full_voxel_file = None
            near_idx_filename = None
            near_pts_filename = None
            full_points = None
            try:
                mesh = trimesh.load(obj_path)
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate([geom for geom in mesh.geometry.values()])

                # 采样点云
                # ========== 【关键】使用 JSON 里的模型真实顶点数 ==========
                num_vertices = shape_info['numVertices']  # 从json读取官方顶点数量
                full_points = mesh.vertices.astype(np.float32)  # 直接用模型顶点，不随机采样
                if num_samples <= 0:
                    num_vertices = shape_info['numVertices']  # 从json读取官方顶点数量
                    points = mesh.vertices.astype(np.float32)  # 直接用模型顶点，不随机采样
                else:
                    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
                    points = points.astype(np.float32)

                # 下采样结果
                pc_centroid = points.mean(axis=0)
                points = points - pc_centroid
                min_gt = min_gt - pc_centroid
                max_gt = max_gt - pc_centroid
                centroid_gt = centroid_gt - pc_centroid

                # 真实点云
                full_points = full_points - pc_centroid

                # 保存
                ## 下采样
                point_filename = f'points_{idx:06d}_{category_id}_{model_id}.npy'
                point_path = os.path.join(point_dir, point_filename)
                np.save(point_path, points)
                point_file = point_filename
                result['point_file'] = point_filename
                result['point_count'] = int(len(points))

                ## 原始点云
                full_point_filename = f'full_points_{idx:06d}_{category_id}_{model_id}.npy'
                full_point_path = os.path.join(point_dir, full_point_filename)
                np.save(full_point_path, full_points)
                full_point_file = full_point_filename
                result['full_point_file'] = full_point_filename
                result['full_point_count'] = int(len(full_points))


            except Exception as e:
                result['errors'].append(f'点云失败: {str(e)}')

            # ========== 处理体素：同样居中 ==========
            voxel_file = None

            try:
                full_voxels = UnifiedShapeNetProcessor._read_binvox(binvox_path)
                sampled_voxels = UnifiedShapeNetProcessor._resize_voxel_32(full_voxels)
                voxel_occupancy = (sampled_voxels > 0).mean() * 100
                # 保存完全体点云
                full_voxel_filename = f'full_voxel_{idx:06d}_{category_id}_{model_id}.npy'
                full_voxel_path = os.path.join(voxel_dir, full_voxel_filename)
                np.save(full_voxel_path, sampled_voxels)
                full_voxel_file = full_voxel_filename
                result['full_voxel_file'] = full_voxel_filename

                # 保存下采样点云
                if voxel_occupancy > 0.1:
                    voxel_filename = f'voxel_{idx:06d}_{category_id}_{model_id}.npy'
                    voxel_path = os.path.join(voxel_dir, voxel_filename)
                    np.save(voxel_path, sampled_voxels)
                    voxel_file = voxel_filename
                    result['voxel_file'] = voxel_filename
                    result['voxel_occupancy'] = float(voxel_occupancy)

                    # ========== 计算最临近映射 ==========
                    bbox_min = min_gt
                    bbox_max = max_gt
                    near_idx, near_pts = UnifiedShapeNetProcessor._compute_voxel_nearest(
                        sampled_voxels, full_points, bbox_min, bbox_max, V=32
                    )

                    near_idx_filename = f'near_idx_{idx:06d}_{category_id}_{model_id}.npy'
                    near_pts_filename = f'near_pts_{idx:06d}_{category_id}_{model_id}.npy'
                    np.save(os.path.join(map_dir, near_idx_filename), near_idx)
                    np.save(os.path.join(map_dir, near_pts_filename), near_pts)

                    result['near_idx_file'] = near_idx_filename
                    result['near_pts_file'] = near_pts_filename
            except Exception as e:
                result['errors'].append(f'体素失败: {str(e)}')

            # ========== 保存元数据 ==========
            if voxel_file and point_file:
                metadata = {
                    'index': idx,
                    'category': category_id,
                    'model_id': model_id,
                    'centroid': centroid_gt.tolist(),
                    'min': min_gt.tolist(),
                    'max': max_gt.tolist(),
                    'voxel_file': voxel_file,
                    'point_file': point_file,
                    'full_point_file': full_point_file,
                    'full_voxel_file': full_voxel_file,
                    'near_idx_file': near_idx_filename,
                    'near_pts_file': near_pts_filename
                }
                meta_fn = f'meta_{idx:06d}_{category_id}_{model_id}.json'
                with open(os.path.join(metadata_dir, meta_fn), 'w') as f:
                    json.dump(metadata, f, indent=2)
                result['metadata_file'] = meta_fn

        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f'错误: {str(e)}')
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
        print(f'   ├─ 临近映射: {self.map_dir}')
        print(f'   └─ 元数据: {self.metadata_dir}')
        print(f'🎯 采样点数: {self.num_samples}')
        print('-' * 80)

        # 准备任务
        tasks = [
            (idx, cat, mid, obj, binvox, json_path,
             str(self.voxel_dir), str(self.point_dir), str(self.map_dir), str(self.metadata_dir),
             self.num_samples)
            for idx, (cat, mid, obj, binvox, json_path) in enumerate(models)
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
                        'full_point_file': result.get('full_point_file'),
                        'full_voxel_file': result.get('full_voxel_file'),
                        'near_idx_file': result.get('near_idx_file'),
                        'near_pts_file': result.get('near_pts_file'),
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
        num_samples=1000
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
