import os

import numpy as np
import open3d as o3d
from joblib import Parallel, delayed
from tqdm import tqdm

from config import config
from src.octree import Octree
from utils.static import get_attention_map


class KittiPrepare:

    def __init__(self, path, seq, trace_parent_num, num_cross_level, pregrouping=False, spatial_len=None,
                 temporal_len=None):
        self.eps = 1e-5
        self.path = os.path.expanduser(path)
        self.seq = seq
        self.trace_parent_num = trace_parent_num
        self.num_cross_level = num_cross_level
        self.bin_files = self.find_and_sort_bin_files(self.path, seq)
        self.horizontal_range = config.kitti.horizontal_range
        self.vertical_range = config.kitti.vertical_range
        horizontal_skeleton = self.horizontal_range
        vertical_skeleton = self.vertical_range
        self.max_depth = config.network.max_octree_level
        self.axis_skeleton = np.array([[horizontal_skeleton, horizontal_skeleton, vertical_skeleton],
                                       [-horizontal_skeleton, -horizontal_skeleton, -vertical_skeleton]])
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 为预分组准备变量 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.spatial_len = spatial_len if spatial_len is not None else config.network.context_len
        self.temporal_len = temporal_len if temporal_len is not None else config.network.refer_len
        self.pregrouping = pregrouping

    def process_single(self, save_dir):
        save_dir = os.path.expanduser(save_dir)

        for i in tqdm(range(1, len(self.bin_files))):
            current_file = self.bin_files[i]
            refer_file = self.bin_files[i - 1]
            # 不是同一个序列
            if os.path.dirname(current_file) != os.path.dirname(refer_file):
                continue
            current_list, current_ancestors_list, refer_list, refer_ancestors_list, attention_map, start_ptr = self.collect_data(
                current_file, refer_file)
            attention_map = np.array(list(attention_map.items()), dtype=np.int64)
            save_path = os.path.join(save_dir, f"{i:06}.npz")
            # 使用 np.savez_compressed 保存这些数组
            np.savez_compressed(save_path,
                                current_list=current_list,
                                current_ancestors_list=current_ancestors_list,
                                refer_list=refer_list,
                                refer_ancestors_list=refer_ancestors_list,
                                attention_map=attention_map,
                                start_ptr=start_ptr)

    def save_data(self, i, current_file, refer_file, save_dir):
        current_list, current_ancestors_list, refer_list, refer_ancestors_list, attention_map, start_ptr = self.collect_data(
            current_file, refer_file)
        attention_map = np.array(list(attention_map.items()), dtype=np.int64)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 数据检查 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        current_invalid = np.where((current_list[:, 4] < 1) | (current_list[:, 4] > 255))[0]
        refer_invalid = np.where((refer_list[:, 4] < 1) | (refer_list[:, 4] > 255))[0]
        if len(current_invalid) > 0 or len(refer_invalid) > 0:
            print(f"{current_file}预处理出现错误，跳过！")
            return
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 保存数据 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        save_path = os.path.join(save_dir, f"{i:06}.npz")
        np.savez_compressed(save_path,
                            current_list=current_list,
                            current_ancestors_list=current_ancestors_list,
                            refer_list=refer_list,
                            refer_ancestors_list=refer_ancestors_list,
                            attention_map=attention_map,
                            start_ptr=start_ptr)

    def process(self, save_dir):
        save_dir = os.path.expanduser(save_dir)

        total_files = len(self.bin_files)
        # 使用 joblib.Parallel 来进行并行计算
        with tqdm(total=total_files - 1) as pbar:
            results = Parallel(n_jobs=-1, return_as='generator_unordered')(  # n_jobs=-1 使用所有可用的 CPU 核心
                delayed(self.save_data)(i, self.bin_files[i], self.bin_files[i - 1], save_dir)
                for i in range(1, total_files)  # 遍历文件列表
                if os.path.dirname(self.bin_files[i]) == os.path.dirname(self.bin_files[i - 1])  # 过滤条件
            )
            # 逐步消费生成器结果，并更新进度条
            for _ in results:
                pbar.update(1)

    def collect_data(self, current_file, refer_file):
        start_ptr = 0
        current_points = np.fromfile(current_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        refer_points = np.fromfile(refer_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 归一化处理 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mask = (current_points[:, 0] >= -self.horizontal_range) & (current_points[:, 0] <= self.horizontal_range) & \
               (current_points[:, 1] >= -self.horizontal_range) & (current_points[:, 1] <= self.horizontal_range) & \
               (current_points[:, 2] >= -self.vertical_range) & (current_points[:, 2] <= self.vertical_range)
        current_points = current_points[mask]
        current_points = np.vstack((current_points, self.axis_skeleton))
        mask = (refer_points[:, 0] >= -self.horizontal_range) & (refer_points[:, 0] <= self.horizontal_range) & \
               (refer_points[:, 1] >= -self.horizontal_range) & (refer_points[:, 1] <= self.horizontal_range) & \
               (refer_points[:, 2] >= -self.vertical_range) & (refer_points[:, 2] <= self.vertical_range)
        refer_points = refer_points[mask]
        refer_points = np.vstack((refer_points, self.axis_skeleton))
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 转换为八叉树 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        current_pcl = o3d.geometry.PointCloud()
        refer_pcl = o3d.geometry.PointCloud()
        current_pcl.points = o3d.utility.Vector3dVector(current_points)
        refer_pcl.points = o3d.utility.Vector3dVector(refer_points)
        current_octree = o3d.geometry.Octree(max_depth=self.max_depth)
        current_octree.convert_from_point_cloud(current_pcl, size_expand=1e-4)
        refer_octree = o3d.geometry.Octree(max_depth=self.max_depth)
        refer_octree.convert_from_point_cloud(refer_pcl, size_expand=1e-4)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 遍历得数据 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        current_octree = Octree(current_octree, self.trace_parent_num, 10000)
        refer_octree = Octree(refer_octree, self.trace_parent_num, 10000)
        current_list = current_octree.octree_data
        current_ancestors_list = current_octree.ancestors_data
        refer_list = refer_octree.octree_data
        refer_ancestors_list = refer_octree.ancestors_data
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 查找起始点 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        for i in range(self.spatial_len):
            window = current_list[i:i + self.spatial_len]
            mask = np.isin(window[:, 0], window[:, 1])
            if not np.any(mask):
                start_ptr = i + 1
                break
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 预先分组加速 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self.pregrouping:
            node_num = current_list.shape[0]
            need_recorded = []
            #      ┌──────────────────────────────────────────────────────────┐
            #      │         注意这里start_ptr为开始编码的第一位index              │
            #      └──────────────────────────────────────────────────────────┘

            for i in range(start_ptr, node_num, self.spatial_len):
                if i < node_num:
                    need_recorded.append(current_list[i - 1][0])

            need_recorded = np.array(need_recorded)
            for i in range(self.max_depth):
                need_recorded = need_recorded[need_recorded >= 1]
                parent_mask = np.isin(current_list[:, 0], need_recorded)
                parent_ids = current_list[parent_mask, 1]
                need_recorded = np.concatenate((need_recorded, parent_ids))
                need_recorded = np.unique(need_recorded, axis=0)
            need_recorded = np.sort(need_recorded, axis=0)
            attention_list = current_list[need_recorded]
        # 不使用预分组
        else:
            attention_list = current_list
        attention_map = get_attention_map(attention_list, refer_octree, self.num_cross_level)
        return current_list, current_ancestors_list, refer_list, refer_ancestors_list, attention_map, start_ptr

    def find_and_sort_bin_files(self, scan_dir, sequences):
        """
        查找并对所有的*.bin文件进行排序
        :return: List(file_path)
        """

        def find_bin_files():
            files = []
            for seq in sequences:
                # to string
                seq = '{0:02d}'.format(int(seq))
                # get paths for each
                scan_path = os.path.join(scan_dir, seq, "velodyne")
                if not os.path.exists(scan_path):
                    raise RuntimeError('Path {} does not exist'.format(scan_path))
                # get files
                scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(scan_path)) for f in fn if f.endswith('.bin')]
                files.extend(scan_files)
            return files

        # 使用glob找到所有的*.bin文件
        bin_files = find_bin_files()

        # 定义一个函数来提取父目录和文件名的信息
        def sort_key(filepath):
            velodyne_dir = os.path.dirname(filepath)
            parent_dir = os.path.basename(os.path.dirname(velodyne_dir))
            filename = os.path.splitext(os.path.basename(filepath))[0]
            return (int(parent_dir), int(filename))

        # 按父目录和文件名进行排序
        sorted_bin_files = sorted(bin_files, key=sort_key)
        return sorted_bin_files
