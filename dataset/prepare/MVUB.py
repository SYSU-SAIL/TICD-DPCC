import gc
import glob
import os.path

import numpy as np
import open3d as o3d
from joblib import Parallel, delayed
from tqdm import tqdm

from config import config
from src.octree import Octree
from utils.static import get_attention_map


class MVUBPrepare:
    def __init__(self, path, seq, trace_parent_num, num_cross_level, pregrouping=False, spatial_len=None,
                 temporal_len=None):
        self.eps = 1e-5
        self.path = os.path.expanduser(path)
        self.seq = seq
        self.trace_parent_num = trace_parent_num
        self.num_cross_level = num_cross_level
        self.pregrouping = pregrouping
        self.spatial_len = spatial_len
        self.temporal_len = temporal_len
        self.resolution_range = config.dense.resolution_range
        self.offset = config.dense.offset
        self.max_depth = config.network.max_octree_level
        self.axis_skeleton = np.array([[0, 0, 0],
                                       [self.resolution_range] * 3])
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 为预分组准备变量 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.spatial_len = spatial_len if spatial_len is not None else config.network.context_len
        self.temporal_len = temporal_len if temporal_len is not None else config.network.refer_len
        self.pregrouping = pregrouping
        self.ply_files = self.find_and_sort_files()

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

        total_files = len(self.ply_files)
        # 使用 joblib.Parallel 来进行并行计算
        with tqdm(total=total_files - 1) as pbar:
            results = Parallel(n_jobs=-1, return_as='generator_unordered')(  # n_jobs=-1 使用所有可用的 CPU 核心
                delayed(self.save_data)(i, self.ply_files[i], self.ply_files[i - 1], save_dir)
                for i in range(1, total_files)  # 遍历文件列表
                if os.path.dirname(self.ply_files[i]) == os.path.dirname(self.ply_files[i - 1])  # 过滤条件
            )
            # 逐步消费生成器结果，并更新进度条
            for _ in results:
                pbar.update(1)
                del _
                gc.collect()  # 手动触发垃圾回收

    def collect_data(self, current_file, refer_file):
        start_ptr = 0
        current_pcd = o3d.io.read_point_cloud(current_file)
        refer_pcd = o3d.io.read_point_cloud(refer_file)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 添加骨架和偏移 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        current_point = np.asarray(current_pcd.points) + 0.5
        current_point = np.vstack((current_point, self.axis_skeleton))
        current_point = current_point + self.offset
        refer_point = np.asarray(refer_pcd.points) + 0.5
        refer_point = np.vstack((refer_point, self.axis_skeleton))
        refer_point = refer_point + self.offset
        current_pcd.points = o3d.utility.Vector3dVector(current_point)
        refer_pcd.points = o3d.utility.Vector3dVector(refer_point)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 转换为八叉树 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        current_octree = o3d.geometry.Octree(max_depth=self.max_depth)
        current_octree.convert_from_point_cloud(current_pcd, size_expand=1e-5)
        refer_octree = o3d.geometry.Octree(max_depth=self.max_depth)
        refer_octree.convert_from_point_cloud(refer_pcd, size_expand=1e-5)
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

    def find_and_sort_files(self):
        root_dir = self.path
        ply_file_path = []
        for seq in self.seq:
            ply_file_dir = os.path.join(root_dir, seq, 'ply')
            # 获取所有 .ply 文件路径
            ply_files = glob.glob(os.path.join(ply_file_dir, "*.ply"))
            # 按文件名中的数字部分排序
            ply_files.sort(key=lambda x: int(os.path.basename(x).replace("frame", "").replace(".ply", "")))
            ply_file_path = ply_file_path + ply_files
        return ply_file_path
