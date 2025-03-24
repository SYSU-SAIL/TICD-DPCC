from ctypes import c_bool, c_double
from multiprocessing import Manager, Array, Pool, shared_memory

import numpy as np
import open3d as o3d
import time
from collections import deque, OrderedDict

from concurrent.futures import ProcessPoolExecutor, as_completed

from joblib import Parallel, delayed, cpu_count
from numpy import ndarray
from tqdm import tqdm

from utils.octant_tools import leaf_to_symbol


class LRUCache:
    def __init__(self, max_size=512):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            # 将最近使用的条目移到末尾
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            # 更新并移动到末尾
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # 删除最旧的条目
            self.cache.popitem(last=False)


class Octree:
    def __init__(self, octree: o3d.geometry.Octree):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 使用队列进行广度优先搜索(BFS) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.octree_data = []  # [idx,parent,depth,octant,symbol]
        idx = 0
        #                              idx | parent | depth | octant
        #                               │      │        │       │
        #                               └──┐   │   ┌────┘ ┌─────┘
        #                                  │   │   │      │
        queue = deque([(octree.root_node, idx, -1, 0, 0)])  # (node, idx, parent,depth)
        idx += 1
        while queue:
            node_obj, node_idx, node_parent_idx, node_depth, node_octant = queue[0]
            queue.popleft()
            # 数据保存进入octree属性
            node_symbol = leaf_to_symbol(node_obj)
            self.octree_data.append([node_idx, node_parent_idx, node_depth, node_octant, node_symbol])
            # 字节点push进队列
            if isinstance(node_obj, o3d.geometry.OctreeInternalNode):
                for octant, child in enumerate(node_obj.children):
                    if child is not None and isinstance(child, o3d.geometry.OctreeInternalNode):
                        queue.append((child, idx, node_idx, node_depth + 1, octant))
                        idx += 1
        self.octree_data = np.array(self.octree_data, dtype=int)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 使用白板 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.padded_leaf = np.full((8, 5), -1)
        self.padded_node = np.full((5), -1)
        self.padded_leaf[:, 3] = np.arange(8)
        self.node_num = self.octree_data.shape[0]
        self.current_index = 0
        # 使用 LRU 缓存，设置缓存大小限制
        self.children_cache = LRUCache(max_size=512)
        self.n_children_cache = LRUCache(max_size=512)
        self.n_parent_cache = LRUCache(max_size=512)
        self.n_children_parent = LRUCache(max_size=512)
        self.parent_list_cache = LRUCache(max_size=512)

    def get_node_by_id(self, idx: int) -> ndarray:
        """
        根据id检索节点
        :param idx: 节点id
        :return: 节点 [idx,parent,depth,octant,symbol]
        """
        node_row = self.octree_data[self.octree_data[:, 0] == idx][0]
        return node_row

    def get_children_of_id(self, idx: int, empty_padding=False):
        """
        根据id获得叶子节点
        :param idx: 节点id
        :param empty_padding: 是否填充0数值
        :return: 多个节点 [[idx,parent,depth,octant,symbol]]
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 读取缓存 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cache_key = (idx, empty_padding)
        cached_result = self.children_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 无缓读取 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if not empty_padding:
            child_rows = self.octree_data[self.octree_data[:, 1] == idx] if idx >= 0 else self.padded_leaf.copy()
        else:
            child_rows = self.padded_leaf.copy()
            # 查找每个 A 中的第二列在 B 中的索引
            if idx >= 0:
                raw_rows = self.octree_data[self.octree_data[:, 1] == idx]
                indices = np.searchsorted(child_rows[:, 3], raw_rows[:, 3])
                child_rows[indices] = raw_rows
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 缓存查询结果 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.children_cache.put(cache_key, child_rows)
        return child_rows

    def get_parent_of_id(self, idx: int):
        """
        根据id查找父节点
        :param idx: 节点id
        :return: 父节点 [idx,parent,depth,octant,symbol]
        """
        if idx < 0:
            return self.padded_node.copy()
        parent_idx = self.octree_data[self.octree_data[:, 0] == idx][0][1]
        if parent_idx >= 0:
            parent_row = self.octree_data[self.octree_data[:, 0] == parent_idx][0]
        else:
            parent_row = self.padded_node.copy()
        return parent_row

    def get_n_parent_of_id(self, idx: int, n: int = 1):
        """
        根据id查找n阶父节点
        :param idx: 节点id
        :param n: 阶数n
        :return: n阶父节点 [idx,parent,depth,octant,symbol]
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 缓存器 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cache_key = (idx, n)
        cached_result = self.n_parent_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 无缓读取 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        for i in range(n):
            target_node = self.octree_data[self.octree_data[:, 0] == idx]
            if len(target_node) != 0:
                idx = self.octree_data[self.octree_data[:, 0] == idx][0][1]
            else:
                return None
        parent_row = self.get_node_by_id(idx)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 缓存查询结果 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.n_parent_cache.put(cache_key, parent_row)
        return parent_row

    def get_parent_list_of_id(self, idx: int, n: int = 1):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 查看缓存 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cache_key = (idx, n)
        cached_result = self.parent_list_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 无缓存处理 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        parent_list = []
        idx = idx
        for i in range(n):
            parent_node = self.get_parent_of_id(idx)
            idx = parent_node[0]
            parent_list.append(parent_node)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 注册缓存 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.parent_list_cache.put(cache_key, parent_list)
        return parent_list

    def get_n_children_of_id(self, idx: int, n: int = 1, empty_padding=False):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 缓存器 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        cache_key = (idx, n, empty_padding)
        cached_result = self.n_children_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 数据读取 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        begin_node = self.get_node_by_id(idx)
        parent_list = np.array([begin_node])
        child_list = np.zeros((0, 5), dtype=int)

        for i in range(n):
            for parent_node in parent_list:
                children = self.get_children_of_id(parent_node[0], empty_padding)
                child_list = np.append(child_list, children, axis=0)
            parent_list = child_list
            child_list = np.zeros((0, 5), dtype=int)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 缓存查询结果 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.n_children_cache.put(cache_key, parent_list)
        return parent_list

    def get_context_window(self, idx: int, context_len: int = 512, trace_parent_num: int = 3):
        half_len = context_len // 2
        start_index = idx - half_len
        end_index = idx + half_len
        if start_index < 0:
            window = self.octree_data[0:context_len, :]
        elif end_index >= self.octree_data.shape[0]:
            window = self.octree_data[-context_len:, :]
        else:
            window = self.octree_data[start_index:start_index + context_len, :]
        parent_list = []
        for row in window:
            idx = row[0]
            parent_node = self.get_parent_list_of_id(idx, trace_parent_num)
            parent_list.append(parent_node)
        parent_list = np.array(parent_list)
        return window, parent_list

    def get_n_children_parent_of_id(self, idx: int, n: int = 1, empty_padding=False, trace_parent_num=3):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 缓存器 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        parent_cache_key = (idx, n, empty_padding, trace_parent_num)
        parent_cached_result = self.n_children_parent.get(parent_cache_key)
        child_cache_key = (idx, n, empty_padding)
        child_cached_result = self.n_children_cache.get(child_cache_key)
        if parent_cache_key is not None and child_cached_result is not None:
            return child_cached_result, parent_cached_result
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 数据读取 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        child_list = self.get_n_children_of_id(idx, n, empty_padding)
        parent_list = []
        for row in child_list:
            idx = row[0]
            parent_node = self.get_parent_list_of_id(idx, trace_parent_num)
            parent_list.append(parent_node)
        parent_list = np.array(parent_list)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 缓存查询结果 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.n_children_parent.put(parent_cache_key, parent_list)
        return child_list, parent_list

    def get_root_node(self):
        """
        获得根节点
        :return: 根节点 [idx,parent,depth,octant,symbol]
        """
        root_node = self.octree_data[self.octree_data[:, 2] == 0][0]
        return root_node

    def get_all_nodes(self):
        """
        获得所有节点
        :return: 所有节点 [[idx,parent,depth,octant,symbol]]
        """
        return self.octree_data

    def __iter__(self):
        self.current_index = 0  # 重置索引
        return self

    def __next__(self):
        if self.current_index < self.node_num:
            node = self.octree_data[self.current_index]
            self.current_index += 1
            return node
        else:
            raise StopIteration


class Octree2Points:
    def __init__(self, min_bound: np.ndarray, max_bound: np.ndarray, depth: int):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.depth = depth
        self.bound_affinity = np.array([
            [0, 0, 0],  # 子叶 0，000
            [0, 0, 1],  # 子叶 1，001
            [0, 1, 0],  # 子叶 2，010
            [0, 1, 1],  # 子叶 3，011
            [1, 0, 0],  # 子叶 4，100
            [1, 0, 1],  # 子叶 5，101
            [1, 1, 0],  # 子叶 6，110
            [1, 1, 1],  # 子叶 7，111
        ])
        # child_min_bounds = min_bound + (bound_affinity * (max_bound - min_bound) / 2)
        # child_max_bounds = min_bound + ((1 - bound_affinity) * (max_bound - min_bound) / 2)

    @staticmethod
    def decode_node(node, octree, max_bound, min_bound, depth, bound_affinity):
        child_id = None
        child_leaf_min_bounds = None
        child_leaf_max_bounds = None
        node_id = node[0]
        current_depth = node[2]
        if current_depth == depth - 1 or node_id < 0:
            return None, None, None, None
        child_min_bounds = min_bound + (bound_affinity * (max_bound - min_bound) / 2)
        child_max_bounds = child_min_bounds + (max_bound - min_bound) / 2
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 查找子节点 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        child_leaf = octree[octree[:, 1] == node_id]

        if len(child_leaf) > 0:
            child_octant = child_leaf[:, 3]
            child_id = child_leaf[:, 0]
            child_leaf_min_bounds = child_min_bounds[child_octant, :]
            child_leaf_max_bounds = child_max_bounds[child_octant, :]
        return node_id, child_id, child_leaf_min_bounds, child_leaf_max_bounds

    def parally_convert(self, octree: np.ndarray):
        node_len = octree.shape[0]
        min_bounds_list = np.zeros((node_len, 3))
        max_bounds_list = np.zeros((node_len, 3))
        ready4decode = np.zeros(node_len, dtype=bool)
        min_bounds_list[0] = self.min_bound
        max_bounds_list[0] = self.max_bound
        ready4decode[0] = True
        decoded_list = np.zeros(node_len, dtype=bool)
        n_jobs = cpu_count()
        inited_size = 3

        def producer():
            while True:
                if decoded_list.all():
                    return
                wait4decode = ~decoded_list
                can_decode_list = wait4decode & ready4decode
                if can_decode_list.any():
                    can_decode_indices = octree[can_decode_list]
                    decoded_list[can_decode_list] = True
                    yield can_decode_indices

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 非叶子节点解码 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with tqdm(total=node_len) as pbar:
            with Parallel(n_jobs=n_jobs, return_as='generator_unordered') as parallel:
                for nodes in producer():
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 满足BATCH条件 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if nodes.shape[0] > inited_size:
                        results = parallel(delayed(self.decode_node)(node, octree, max_bounds_list[node[0]],
                                                                     min_bounds_list[node[0]], self.depth,
                                                                     self.bound_affinity) for node in nodes)

                        for node_id, child_id, child_leaf_min_bounds, child_leaf_max_bounds in results:
                            if child_id is not None:
                                min_bounds_list[child_id, :] = child_leaf_min_bounds
                                max_bounds_list[child_id, :] = child_leaf_max_bounds
                                ready4decode[child_id] = True
                            pbar.update(1)
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 不满足BATCH条件 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    else:
                        for node in nodes:
                            node_id = node[0]
                            node_id, child_id, child_leaf_min_bounds, child_leaf_max_bounds = self.decode_node(node,
                                                                                                               octree,
                                                                                                               max_bounds_list[
                                                                                                                   node_id],
                                                                                                               min_bounds_list[
                                                                                                                   node_id],
                                                                                                               self.depth,
                                                                                                               self.bound_affinity)
                            if child_id is not None:
                                min_bounds_list[child_id, :] = child_leaf_min_bounds
                                max_bounds_list[child_id, :] = child_leaf_max_bounds
                                ready4decode[child_id] = True
                            pbar.update(1)

        @staticmethod
        def decode_leaf_node(node, max_bound, min_bound, bound_affinity):
            child_min_bounds = min_bound + (bound_affinity * (max_bound - min_bound) / 2)
            child_max_bounds = child_min_bounds + (max_bound - min_bound) / 2
            symbol = node[-1]
            bool_mask = [(symbol >> i) & 1 for i in range(7, -1, -1)]  # 从高位到低位提取每一位
            bool_mask = np.array(bool_mask).astype(bool)
            point = (child_min_bounds[bool_mask, :] + child_max_bounds[bool_mask, :]) / 2
            return point.tolist()

        selected_leaf = octree[octree[:, 2] == self.depth - 1]

        point_output = []
        with tqdm(total=selected_leaf.shape[0]) as pbar:
            results = Parallel(n_jobs=-1, return_as='generator_unordered')(  # n_jobs=-1 使用所有可用的 CPU 核心
                delayed(decode_leaf_node)(leaf, max_bounds_list[leaf[0]], min_bounds_list[leaf[0]],
                                          self.bound_affinity) for leaf in selected_leaf  # 过滤条件
            )
            # 逐步消费生成器结果，并更新进度条
            for res in results:
                point_output.extend(res)
                pbar.update(1)
        point_output = np.array(point_output)
        point_output = point_output[:, [2, 1, 0]]
        return point_output

    def convert(self, octree: np.ndarray):
        node_len = octree.shape[0]
        min_bounds = np.zeros((node_len, 3), dtype=float)
        max_bounds = np.zeros((node_len, 3), dtype=float)
        min_bounds[0] = self.min_bound
        max_bounds[0] = self.max_bound

        for i in range(0, node_len):
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 计算边界框 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            row = octree[i]
            id = row[0]
            current_depth = row[2]
            if current_depth == self.depth - 1:
                break
            current_max_bound = max_bounds[i]
            current_min_bound = min_bounds[i]
            child_min_bounds = current_min_bound + (self.bound_affinity * (current_max_bound - current_min_bound) / 2)
            child_max_bounds = child_min_bounds + (current_max_bound - current_min_bound) / 2
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 查找子节点 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            child_leaf = octree[octree[:, 1] == id]
            if len(child_leaf) > 0:
                child_octant = child_leaf[:, 3]
                child_id = child_leaf[:, 0]
                child_leaf_min_bounds = child_min_bounds[child_octant, :]
                child_leaf_max_bounds = child_max_bounds[child_octant, :]
                min_bounds[child_id, :] = child_leaf_min_bounds
                max_bounds[child_id, :] = child_leaf_max_bounds

        selected_leaf = octree[octree[:, 2] == self.depth - 1]
        selected_id = selected_leaf[:, 0]
        selected_min_bounds = min_bounds[selected_id, :]
        selected_max_bounds = max_bounds[selected_id, :]
        # 最后一级要单独解码
        point_output = np.empty((0, 3))
        for i, row in enumerate(selected_leaf):
            min_bound = selected_min_bounds[i]
            max_bound = selected_max_bounds[i]
            child_min_bounds = min_bound + (self.bound_affinity * (max_bound - min_bound) / 2)
            child_max_bounds = child_min_bounds + (max_bound - min_bound) / 2
            symbol = row[-1]
            bool_mask = [(symbol >> i) & 1 for i in range(7, -1, -1)]  # 从高位到低位提取每一位
            bool_mask = np.array(bool_mask).astype(bool)
            point = (child_min_bounds[bool_mask, :] + child_max_bounds[bool_mask, :]) / 2
            point_output = np.vstack([point_output, point])
        point_output = point_output[:, [2, 1, 0]]
        return point_output
