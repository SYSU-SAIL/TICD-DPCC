import glob
import multiprocessing
import os.path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
from config import config


class OctNodeDDP(IterableDataset):
    def __init__(self, root, rank, world_size, data_type="train", max_depth=None, spatial_len=128, temporal_len=256,
                 sample_rate=-1, depth_decap={}):
        self.root = os.path.expanduser(root)
        self.sample_rate = sample_rate
        self.type = data_type
        self.file_dir = os.path.join(self.root, self.type)
        self.file_path = []
        for root, dirs, files in os.walk(self.file_dir):
            for file in files:
                if file.endswith('.npz'):
                    self.file_path.append(os.path.join(root, file))
        self.file_path = sorted(self.file_path)
        if self.sample_rate > 0:
            num_samples = int(len(self.file_path) * sample_rate)
            indices = np.linspace(0, len(self.file_path) - 1, num_samples, dtype=int)
            sample_files = [self.file_path[i] for i in indices]
            self.file_path = sample_files
        self.spatial_len = spatial_len
        self.temporal_len = temporal_len
        self.half_window = (temporal_len - spatial_len) // 2
        self.file_num = len(self.file_path)
        self.depth_decap = depth_decap
        self.max_depth = max_depth if max_depth is not None else config.network.octree_level
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 实现DDP部分 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.rank = rank
        self.world_size = world_size


    def __iter__(self):
        for file_index, file_path in enumerate(self.file_path):
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 确保多WORKER的正确性 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            worker_id = 0  # 默认是单线程
            num_workers = 1  # 默认是单线程
            worker_info = get_worker_info()
            if worker_info is not None:
                # 多线程运行时获取 worker ID
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
            # 确保是当前线程的工作
            world_worker_id = self.rank * num_workers + worker_id
            if file_index % (self.world_size * num_workers) != world_worker_id:
                continue
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 加一个数量限制机制 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            depth_decap_dict = {i: 0 for i in range(self.max_depth)}
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 读取数据 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            data = np.load(file_path)
            start_ptr = data['start_ptr']
            current_list = data['current_list']
            depth_mask = current_list[:, 2] < self.max_depth
            current_list = current_list[depth_mask]
            current_ancestors_list = data['current_ancestors_list']
            current_ancestors_list = current_ancestors_list[depth_mask]
            refer_list = data['refer_list']
            depth_mask = refer_list[:, 2] < self.max_depth
            refer_list = refer_list[depth_mask]
            refer_ancestors_list = data['refer_ancestors_list']
            refer_ancestors_list = refer_ancestors_list[depth_mask]
            attention_map = data['attention_map']
            root_attention = np.array([[0, 0]])
            attention_map = np.vstack([root_attention, attention_map])
            refer_len = refer_list.shape[0]
            node_num = current_list.shape[0]
            for i in range(start_ptr, node_num, self.spatial_len):
                if i + self.spatial_len < node_num:
                    current_clip = current_list[i:i + self.spatial_len]
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 对应深层进行采样 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if np.all(current_clip[:, 2] == current_clip[0, 2]):
                        clip_depth = current_clip[0, 2]
                        if clip_depth in self.depth_decap.keys():
                            depth_decap_dict[clip_depth] += 1
                            if depth_decap_dict[clip_depth] % self.depth_decap[clip_depth] != 0:
                                continue
                    current_ancestors_clip = current_ancestors_list[i:i + self.spatial_len]
                    search_idx = current_list[i - 1][0]  # 上一个编码序列的最后一位
                    attetnion_anchor = int(attention_map[attention_map[:, 0] == search_idx, 1])
                    start = max(0, attetnion_anchor - self.half_window)
                    end = min(refer_len, attetnion_anchor + self.spatial_len + self.half_window)
                    # 如果窗口还不够2048的大小，调整起始和结束
                    if start == 0:
                        end = min(self.temporal_len, refer_len)
                    elif end == refer_len:
                        start = max(refer_len - self.temporal_len, 0)
                    refer_clip = refer_list[start:end]
                    refer_ancestors_clip = refer_ancestors_list[start:end]
                    label = current_clip[:, 4].copy()
                    # 作为Padded
                    current_clip[:, 4] = 0
                    # 将所有-1 padded为0
                    current_clip[current_clip == -1] = 0
                    current_ancestors_clip[current_ancestors_clip == -1] = 0
                    refer_clip[refer_clip == -1] = 0
                    refer_ancestors_clip[refer_ancestors_clip == -1] = 0
                    data = {"current_clip": current_clip, "current_ancestors_clip": current_ancestors_clip,
                            "refer_clip": refer_clip, "refer_ancestors_clip": refer_ancestors_clip, "label": label,
                            'file_index': file_index}
                    yield data
