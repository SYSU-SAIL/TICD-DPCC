from collections import deque
from copy import deepcopy, copy

import open3d as o3d


def leaf_to_symbol(node: o3d.geometry.OctreeInternalNode):
    """
    当前节点的占用情况转为符号
    :param node: 当前节点
    :return: 符号[0-255]
    """
    # 获取 node 的 children 列表
    children = node.children
    # 创建一个二进制字符串
    binary_string = ''.join(['1' if child is not None else '0' for child in children])
    # 将二进制字符串转换为整数
    res = int(binary_string, 2)
    return res


def get_clidren_symbol_list(node: o3d.geometry.OctreeInternalNode, assigned_level=None):
    """
    获得子节点的占用符号表示
    :param node: 当前节点
    :param assigned_level: 为获得的数据标记level（depth）
    :return: [symbol,(level),oct]
    """
    symbol_list = []
    for idx, child in enumerate(node.children):
        if isinstance(child, o3d.geometry.OctreeInternalNode):
            symbol_list.append(
                [leaf_to_symbol(child), idx] if assigned_level is None else [leaf_to_symbol(child), assigned_level,
                                                                             idx])
        else:
            symbol_list.append([0, idx] if assigned_level is None else [0, assigned_level, idx])
    return symbol_list


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class OctRebuild():
    def __init__(self, trace_parent_num=3):
        self.trace_parent_num = trace_parent_num
        self.octree_data = []  # 存储 [id, parent_id, depth, octant, symbol]
        self.ancestors_data = []  # 存储祖先信息 [N, trace_parent_num, 5]
        # 初始化队列 (node_idx, parent_idx, depth, octant, ancestors)
        self.queue = deque([(0, -1, 0, 0, [[-1] * 5] * trace_parent_num)])  # 根节点
        self.idx = 1  # 从 1 开始分配 ID，根节点 ID 为 0

    def rebuild(self, symbols):
        for symbol in symbols:
            # 出队一个节点
            node_idx, parent_idx, depth, octant, ancestors = self.queue.popleft()

            # 保存当前节点的属性
            self.octree_data.append([node_idx, parent_idx, depth, octant, symbol])

            # 更新祖先列表
            current_ancestors = ancestors[-self.trace_parent_num:]
            self.ancestors_data.append(current_ancestors)

            # 解析符号并添加子节点
            binary_str = f"{symbol:08b}"  # 将符号转为二进制表示
            for child_octant, bit in enumerate(binary_str):
                if bit == '1':  # 子节点被占用
                    self.queue.append((self.idx, node_idx, depth + 1, child_octant,
                                       current_ancestors[-(self.trace_parent_num - 1):] + [
                                           [node_idx, parent_idx, depth, octant, symbol]]))
                    self.idx += 1

    def per_build(self,node_number):
        symbol = 0
        queue_copy = deque(self.queue)  # 浅拷贝队列
        octree_data = list(self.octree_data)  # 浅拷贝数据
        ancestors_data = list(self.ancestors_data)  # 浅拷贝祖先数据
        idx = copy(self.idx)
        for i in range(node_number):
            # 出队一个节点
            node_idx, parent_idx, depth, octant, ancestors = queue_copy.popleft()

            # 保存当前节点的属性
            octree_data.append([node_idx, parent_idx, depth, octant, symbol])

            # 更新祖先列表
            current_ancestors = ancestors[-self.trace_parent_num:]
            ancestors_data.append(current_ancestors)

            # 解析符号并添加子节点
            binary_str = f"{symbol:08b}"  # 将符号转为二进制表示
            for child_octant, bit in enumerate(binary_str):
                if bit == '1':  # 子节点被占用
                    queue_copy.append((idx, node_idx, depth + 1, child_octant,
                                       current_ancestors[-(self.trace_parent_num - 1):] + [
                                           [node_idx, parent_idx, depth, octant, symbol]]))
                    idx += 1
        return octree_data, ancestors_data