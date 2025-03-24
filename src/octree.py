import octree_module

import time
from collections import deque, OrderedDict

import numpy as np
import open3d as o3d
from numpy import ndarray

from utils.octant_tools import leaf_to_symbol


class Octree:
    def __init__(self, octree: o3d.geometry.Octree, trace_parent_num, cache_size):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 使用队列进行广度优先搜索(BFS) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.octree_data = []  # [idx,parent,depth,octant,symbol]
        self.ancestors_data = []  # [N, trace_parent_num, 5]
        idx = 0
        #                              idx | parent | depth | octant
        #                               │      │        │       │
        #                               └──┐   │   ┌────┘ ┌─────┘
        #                                  │   │   │      │
        queue = deque([(octree.root_node, idx, -1, 0, 0, [[-1] * 5] * trace_parent_num)])  # (node, idx, parent,depth)
        idx += 1
        while queue:
            node_obj, node_idx, node_parent_idx, node_depth, node_octant, ancestors = queue[0]
            queue.popleft()
            # 数据保存进入octree属性
            node_symbol = leaf_to_symbol(node_obj)
            self.octree_data.append([node_idx, node_parent_idx, node_depth, node_octant, node_symbol])

            # 更新祖先列表并截取最新的 trace_parent_num 个祖先
            self.ancestors_data.append(ancestors)
            current_ancestors = ancestors[-(trace_parent_num - 1):] + [
                [node_idx, node_parent_idx, node_depth, node_octant, node_symbol]]

            # 字节点push进队列
            if isinstance(node_obj, o3d.geometry.OctreeInternalNode):
                for octant, child in enumerate(node_obj.children):
                    if child is not None and isinstance(child, o3d.geometry.OctreeInternalNode):
                        queue.append((child, idx, node_idx, node_depth + 1, octant, current_ancestors))
                        idx += 1
        self.octree_data = np.array(self.octree_data, dtype=int)
        self.ancestors_data = np.array(self.ancestors_data, dtype=int)  # 转换为 ndarray
        self.octree = octree_module.Octree(self.octree_data, self.ancestors_data, cache_size)

    def get_node_by_id(self, idx: int) -> ndarray:
        """
        根据id检索节点
        :param idx: 节点id
        :return: 节点 [idx,parent,depth,octant,symbol]
        """
        return self.octree.get_node_by_id(idx)

    def get_children_of_id(self, idx: int, empty_padding=False):
        """
        根据id获得叶子节点
        :param idx: 节点id
        :param empty_padding: 是否填充0数值
        :return: 多个节点 [[idx,parent,depth,octant,symbol]]
        """
        return self.octree.get_children_of_id(idx, empty_padding)

    def get_parent_of_id(self, idx: int):
        """
        根据id查找父节点
        :param idx: 节点id
        :return: 父节点 [idx,parent,depth,octant,symbol]
        """
        return self.octree.get_parent_of_id(idx)

    def get_n_parent_of_id(self, idx, n):
        return self.octree.get_n_parent_of_id(idx, n)

    def get_parent_list_of_id(self, idx: int):

        return self.octree.get_parent_list_of_id(idx)

    def get_n_children_of_id(self, idx: int, n: int = 1, empty_padding=False):

        return self.octree.get_n_children_of_id(idx, n, empty_padding)

    def get_context_window(self, idx: int, context_len: int = 512):

        return self.octree.get_context_window(idx, context_len)

    def get_root_node(self):
        """
        获得根节点
        :return: 根节点 [idx,parent,depth,octant,symbol]
        """
        return self.octree.get_root_node()

    def get_all_nodes(self):
        """
        获得所有节点
        :return: 所有节点 [[idx,parent,depth,octant,symbol]]
        """
        return self.octree_data

    def preview_next(self):
        return self.octree.preview_next()

    def __iter__(self):
        return self.octree.__iter__()

    def __next__(self):
        return self.octree.__next__()
