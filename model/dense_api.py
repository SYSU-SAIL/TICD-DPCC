import collections
import os.path
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from config import config
from model.recursive_attention import RecursiveAttention
from numpyAc import numpyAc
from src.octree import Octree
from utils.octant_tools import OctRebuild
from utils.static import get_attention_map

EXTENSION = ['ply']


class DenseAPI(object):
    def __init__(self, model_path, static_prob_file, depth, batch_size=32):
        self.encoder_buffer = None
        self.decoder_buffer = None
        self.tree_buffer = None
        self.eps = 1e-5
        self.batch_size = batch_size
        self.debug = None
        self.debug1 = None
        self.debug2 = None
        self.debug3 = None
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 整理网络相关配置 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.token_size = config.network.token_size
        self.output_size = config.network.output_size
        self.attention_heads = config.network.attention_heads
        self.context_len = config.network.context_len
        self.temporal_len = config.network.refer_len
        self.anchor_num = config.network.anchor_num
        self.trace_parent_num = config.network.trace_parent_num
        self.num_cross_level = config.network.num_cross_level
        self.model = RecursiveAttention(self.token_size, self.output_size, self.attention_heads, self.trace_parent_num,
                                        self.context_len,
                                        self.anchor_num)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, weights_only=True,map_location=self.device)
        state_prob_data = np.load(static_prob_file)
        self.begin_prob = state_prob_data['begin_prob'].astype(np.float32)
        self.end_prob = state_prob_data['end_prob'].astype(np.float32)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 加载模型 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        model_state_dict = collections.OrderedDict()
        if "model" in state_dict:
            for key, value in state_dict["model"].items():
                if key.startswith("module."):
                    key = key.replace("module.", "")
                model_state_dict.update({key: value})
        self.model = self.model.to(self.device)
        self.model.load_state_dict(model_state_dict)
        torch.backends.mha.set_fastpath_enabled(False)
        self.ancestral_index = self.model.ancestral_index
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 骨架信息 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.resolution_range = config.dense.resolution_range
        self.offset = config.dense.offset
        self.max_depth = depth
        self.axis_skeleton = np.array([[0, 0, 0],
                                       [self.resolution_range] * 3])
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 上下文长度 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.spatial_len = config.network.context_len
        self.temporal_len = config.network.refer_len
        self.half_window = (self.temporal_len - self.spatial_len) // 2
        self.padding_prob = [1 / self.output_size] * self.output_size

    def encode(self, points_pth):
        points_pth = os.path.expanduser(points_pth)
        if os.path.splitext(points_pth)[-1] not in EXTENSION:
            RuntimeError('指定文件拓展名无法读取')
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 读取规范数据 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        pcd = o3d.io.read_point_cloud(points_pth)
        points = np.asarray(pcd.points) + 0.5
        points = np.vstack((points, self.axis_skeleton))
        points = points + self.offset
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 获得八叉树 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        pcd.points = o3d.utility.Vector3dVector(points)
        octree = o3d.geometry.Octree(max_depth=self.max_depth)
        octree.convert_from_point_cloud(pcd,size_expand=1e-5)
        current_octree = Octree(octree, self.trace_parent_num, 10000)
        current_list = current_octree.octree_data.copy()
        oct_seq = current_list[:, 4].copy()
        current_ancestors_list = current_octree.ancestors_data.copy()
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 没有参考帧 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self.encoder_buffer is None:
            self.encoder_buffer = current_octree
            return None, None, None, None
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 获得参考八叉树 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        refer_octree = self.encoder_buffer
        refer_list = refer_octree.octree_data
        refer_ancestors_list = refer_octree.ancestors_data
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 查找起始点 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        start_ptr = 0
        for i in range(self.spatial_len):
            window = current_list[i:i + self.spatial_len]
            mask = np.isin(window[:, 0], window[:, 1])
            if not np.any(mask):
                start_ptr = i + 1
                break
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 注意力分组 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        attention_map = get_attention_map(attention_list, refer_octree, self.num_cross_level)
        attention_map[0] = 0
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 塞到网络 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        refer_len = refer_list.shape[0]
        self.model.eval()
        bit_pro = self.begin_prob[:start_ptr, :].tolist()
        for i in range(start_ptr, node_num, self.spatial_len):
            if i + self.spatial_len < node_num:
                current_clip = current_list[i:i + self.spatial_len]
                current_ancestors_clip = current_ancestors_list[i:i + self.spatial_len]
                search_idx = current_list[i - 1][0]  # 上一个编码序列的最后一位
                attention_anchor = int(attention_map[search_idx])
                start = max(0, attention_anchor - self.half_window)
                end = min(refer_len, attention_anchor + self.spatial_len + self.half_window)
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
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 转为TENSOR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                with torch.no_grad():
                    current_clip = torch.from_numpy(current_clip).unsqueeze(0).unsqueeze(2).to(self.device)
                    current_ancestors_clip = torch.from_numpy(current_ancestors_clip).unsqueeze(0).to(self.device)
                    refer_clip = torch.from_numpy(refer_clip).unsqueeze(0).unsqueeze(2).to(self.device)
                    refer_ancestors_clip = torch.from_numpy(refer_ancestors_clip).unsqueeze(0).to(self.device)
                    label = torch.from_numpy(label).unsqueeze(0).to(self.device)
                    predict_res = self.model(current_clip, current_ancestors_clip, refer_clip, refer_ancestors_clip,
                                             label)
                    outputs = predict_res.view((-1, predict_res.shape[-1]))
                    logits = F.softmax(outputs, dim=1)
                    # 这里需要把分层的两个调整位置
                    logits = self.reorder_with_ancestral_index(logits.detach().cpu().numpy())
                    oct_seq[i:i + self.spatial_len] = self.reorder_with_ancestral_index(
                        oct_seq[i:i + self.spatial_len]).reshape(-1)
                    bit_pro += logits.tolist()
                    pass
            else:
                end_ptr = i
                bit_pro += self.end_prob[-(node_num - end_ptr):,:].tolist()
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 算术编码 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        oct_seq = oct_seq.astype(np.int16) - 1
        bit_pro = np.array(bit_pro, dtype=np.float32)
        arithmetic_coder = numpyAc.arithmeticCoding()
        byte_stream, binsz = arithmetic_coder.encode(bit_pro, oct_seq)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 更新一下BUFFER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.decoder_buffer = self.encoder_buffer
        self.encoder_buffer = current_octree
        return byte_stream, node_num, start_ptr, points

    def decode(self, byte_stream, oct_len, start_ptr):
        decoder = numpyAc.arithmeticDeCoding(byte_stream, oct_len, 255)
        rebuilder = OctRebuild()
        # 解码开头
        freqsinit = self.begin_prob[:start_ptr, :]
        start_seq = decoder.decode(freqsinit)
        rebuilder.rebuild(start_seq)
        refer_octree = self.decoder_buffer
        refer_list = refer_octree.octree_data
        refer_ancestors_list = refer_octree.ancestors_data
        attention_map = {}
        for i in range(start_ptr, oct_len, self.spatial_len):
            if i + self.spatial_len < oct_len:
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 预先生成LIST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                octree_data, ancestors_data = rebuilder.per_build(self.spatial_len)
                octree_data, ancestors_data = np.array(octree_data), np.array(ancestors_data)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 查找相似节点 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                search_node = octree_data[i - 1][0]  # 上一个编码序列的最后一位
                need_recorded = [search_node]
                need_recorded = np.array(need_recorded)
                for j in range(self.max_depth):
                    need_recorded = need_recorded[need_recorded >= 1]
                    parent_mask = np.isin(octree_data[:, 0], need_recorded)
                    parent_ids = octree_data[parent_mask, 1]
                    need_recorded = np.concatenate((need_recorded, parent_ids))
                    need_recorded = np.unique(need_recorded, axis=0)
                need_recorded = np.sort(need_recorded, axis=0)
                attention_list = octree_data[need_recorded]
                attention_map = get_attention_map(attention_list, refer_octree, self.num_cross_level,
                                                  attention_map=attention_map)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 整理输入变量 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                refer_len = refer_list.shape[0]
                attention_anchor = int(attention_map[search_node])
                start = max(0, attention_anchor - self.half_window)
                end = min(refer_len, attention_anchor + self.spatial_len + self.half_window)
                # 如果窗口还不够2048的大小，调整起始和结束
                if start == 0:
                    end = min(self.temporal_len, refer_len)
                elif end == refer_len:
                    start = max(refer_len - self.temporal_len, 0)
                refer_clip = refer_list[start:end]
                refer_ancestors_clip = refer_ancestors_list[start:end]
                current_clip = octree_data[i:i + self.spatial_len]
                current_ancestors_clip = ancestors_data[i:i + self.spatial_len]
                # 将所有-1 padded为0
                current_clip[current_clip == -1] = 0
                current_ancestors_clip[current_ancestors_clip == -1] = 0
                refer_clip[refer_clip == -1] = 0
                refer_ancestors_clip[refer_ancestors_clip == -1] = 0
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 输入网络 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                self.model.eval()
                with torch.no_grad():
                    current_clip = torch.from_numpy(current_clip).unsqueeze(0).unsqueeze(2).to(self.device)
                    current_ancestors_clip = torch.from_numpy(current_ancestors_clip).unsqueeze(0).to(self.device)
                    refer_clip = torch.from_numpy(refer_clip).unsqueeze(0).unsqueeze(2).to(self.device)
                    refer_ancestors_clip = torch.from_numpy(refer_ancestors_clip).unsqueeze(0).to(self.device)
                    # 第一层
                    predict_res = self.model.decode_with_ancestor(current_clip, current_ancestors_clip, refer_clip,
                                                                  refer_ancestors_clip)
                    outputs = predict_res.view((-1, predict_res.shape[-1]))
                    logits = F.softmax(outputs, dim=1)
                    logits = logits.detach().cpu().numpy()
                    decode_first_seq = decoder.decode(logits)
                    decode_first_seq_torch = torch.tensor(decode_first_seq).to(self.device)
                    current_clip[0, self.ancestral_index, 0, -1] = decode_first_seq_torch
                    # 第二层
                    predict_res = self.model.decode_with_sibling(current_clip, current_ancestors_clip, refer_clip,
                                                                 refer_ancestors_clip)
                    outputs = predict_res.view((-1, predict_res.shape[-1]))
                    logits = F.softmax(outputs, dim=1)
                    logits = logits.detach().cpu().numpy()
                    logits = np.delete(logits, self.ancestral_index, axis=0)
                    decode_second_seq = decoder.decode(logits)
                    seq = np.array(decode_first_seq + decode_second_seq)
                    seq = self.restore_original_order(seq, self.spatial_len)
                    rebuilder.rebuild(seq)
            else:
                end_ptr = i
                end_pro = self.end_prob[-(oct_len - end_ptr):, :]
                end_seq = decoder.decode(end_pro)
                rebuilder.rebuild(end_seq)
        octree_data = np.array(rebuilder.octree_data)
        ancestors_data = np.array(rebuilder.ancestors_data)
        return octree_data, ancestors_data

    def reorder_with_ancestral_index(self, sequence):
        """
        将 ancestral_index 指定的元素提前放到序列的开头。

        Args:
            sequence (np.ndarray): 原始序列，形状为 [N, 1]。
            ancestral_index (list[int]): 提前编码部分的索引列表。

        Returns:
            reordered_sequence (np.ndarray): 重排后的序列。
        """
        # 提取提前编码部分
        encoded_part = sequence[self.ancestral_index]

        # 找到剩余未编码的索引
        all_indices = np.arange(len(sequence))
        remaining_indices = list(set(all_indices) - set(self.ancestral_index))
        # 提取剩余未编码部分
        remaining_part = sequence[remaining_indices]
        # 合并：提前编码部分在前，剩余部分在后
        reordered_sequence = np.vstack((encoded_part, remaining_part))
        return reordered_sequence

    def restore_original_order(self, reordered_sequence, original_length):
        """
        将序列恢复到原始顺序。

        Args:
            reordered_sequence (np.ndarray): 重排后的序列，形状为 [N, 1]。
            original_length (int): 原始序列的长度。

        Returns:
            original_sequence (np.ndarray): 恢复到原始顺序的序列。
        """
        # 创建剩余索引列表
        all_indices = np.arange(original_length)
        remaining_indices = list(set(all_indices) - set(self.ancestral_index))
        # 构建原始序列的空数组
        original_sequence = np.zeros_like(reordered_sequence)
        # 提取提前编码部分和剩余部分
        encoded_part = reordered_sequence[:len(self.ancestral_index)]
        remaining_part = reordered_sequence[len(self.ancestral_index):]
        # 根据索引恢复原顺序
        original_sequence[self.ancestral_index] = encoded_part
        original_sequence[remaining_indices] = remaining_part
        return original_sequence
