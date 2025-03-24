import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor
import open3d as o3d
from config import config


def get_attention_map(attention_list, refer_octree, num_cross_level=None, eps=1e-5, attention_map=None):
    attention_map = attention_map if attention_map is not None else {}
    num_cross_level = num_cross_level if num_cross_level is not None else config.network.num_cross_level
    for row in attention_list:
        id, parent, depth, octant, leaf = row[0], row[1], row[2], row[3], row[4]
        if id not in attention_map:
            refer_nodes = None
            if depth == num_cross_level:
                refer_nodes = np.array(refer_octree.get_n_children_of_id(0, num_cross_level),
                                       dtype=np.uint64)

            if depth > num_cross_level:
                refer_id = refer_octree.get_n_parent_of_id(attention_map[parent], num_cross_level - 1)
                refer_nodes = np.array(refer_octree.get_n_children_of_id(refer_id[0], num_cross_level),
                                       dtype=np.uint64)
            if refer_nodes is not None:
                refer_nodes_symbol = refer_nodes[:, 4].astype(np.uint8)
                # 计算交集与并集
                # 跳过负数（padded）的值
                valid_mask = refer_nodes_symbol > 0
                # 仅对有效的（非负的）参考变量进行 IoU 计算
                valid_refs_symbol = np.array(refer_nodes_symbol[valid_mask], dtype=np.uint8)
                intersection = bitwise_count(
                    np.bitwise_and(leaf, valid_refs_symbol, dtype=np.uint8))
                union = bitwise_count(
                    np.bitwise_or(leaf, valid_refs_symbol, dtype=np.uint8))
                iou_scores = intersection / (union + eps)

                parent_ids = refer_nodes[:, 1][valid_mask]
                momentum = np.where(parent_ids == parent, 0.1, 0.0)
                iou_scores += momentum
                # ━━━━━━━━━━━━━━━━━━━━━━━ 找到 IOU 最高的参考变量 ━━━━━━━━━━━━━━━━━━━━━━━
                best_index = np.argmax(iou_scores)
                attention_map[id] = refer_nodes[valid_mask][best_index][0]
    return attention_map


def bitwise_count(tensor):
    # 将 tensor 转换为二进制表示，并统计每个元素中 1 的个数
    # return torch.count_nonzero(tensor.unsqueeze(-1).bitwise_and(torch.tensor([1 << i for i in range(32)])), dim=-1)
    return np.vectorize(lambda x: x.bit_count())(tensor)


def encode_node(logits: Tensor, octvalue: Tensor):
    predicted = torch.argmax(logits, dim=1)  # [N,1]
    selected_values = torch.gather(logits, 1, octvalue.unsqueeze(1)).squeeze(1)
    log2_neg = -torch.log2(selected_values + 1e-07)
    bit_est = torch.mean(log2_neg)
    # 计算预测和实际索引相同的布尔张量
    correct_predictions = torch.eq(predicted, octvalue)
    # 计算正确率
    accuracy = torch.mean(correct_predictions.float())
    return bit_est, accuracy


def psnr(x, max_energy):
    return 10 * np.log10(max_energy / x)


def compute_point_cloud_normal(points):
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points)
    points_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=59.7, max_nn=12))
    # o3d.visualization.draw_geometries([points_o3d], point_show_normal=True)
    normal = np.asarray(points_o3d.normals)
    return normal


def assign_attr(attr1, idx1, idx2):
    """Given point sets x1 and x2, transfers attributes attr1 from x1 to x2.
    idx1: N2 array containing the nearest neighbors indices of x2 in x1
    idx2: N1 array containing the nearest neighbors indices of x1 in x2
    """
    counts = np.zeros(idx1.shape[0])
    attr_sums = np.zeros((idx1.shape[0], attr1.shape[1]))
    for i, idx in enumerate(idx2):
        counts[idx] += 1
        attr_sums[idx] += attr1[i]
    for i, idx in enumerate(idx1):
        if counts[i] == 0:
            counts[i] += 1
            attr_sums[i] += attr1[idx]
    counts = np.expand_dims(counts, -1)
    attr2 = attr_sums / counts
    return attr2


def calc_point_to_point_plane_psnr(points1, points2, r=59.7):
    t = time.time()
    pc_1 = points1[np.where(np.sum(points1, -1) != 0)]
    pc_2 = points2[np.where(np.sum(points2, -1) != 0)]
    t1 = cKDTree(pc_1, balanced_tree=False)
    _, idx1 = t1.query(pc_2)
    t2 = cKDTree(pc_2, balanced_tree=False)
    _, idx2 = t2.query(pc_1)

    max_energy = 3 * r * r
    pc_1_ngb = pc_2[idx2]
    pc_2_ngb = pc_1[idx1]
    point_to_point_mse_1 = np.sum(np.sum((pc_1 - pc_1_ngb) ** 2, axis=1)) / pc_1.shape[0]
    point_to_point_mse_2 = np.sum(np.sum((pc_2 - pc_2_ngb) ** 2, axis=1)) / pc_2.shape[0]
    point_to_point_psnr_1 = psnr(point_to_point_mse_1, max_energy)
    point_to_point_psnr_2 = psnr(point_to_point_mse_2, max_energy)
    point_to_point_result = {
        'psnr_1': point_to_point_psnr_1,
        'psnr_2': point_to_point_psnr_2,
        'mse_1': point_to_point_mse_1,
        'mse_2': point_to_point_mse_2,
        'psnr_mean': (point_to_point_psnr_1 + point_to_point_psnr_2) / 2,
        'mse_mean': (point_to_point_mse_1 + point_to_point_mse_2) / 2,
    }

    pc_1_n = compute_point_cloud_normal(pc_1)
    # pc_2_n = compute_point_cloud_normal(pc_2)
    # Compute normals in pc_2 from normals in pc_1
    pc_2_n = assign_attr(pc_1_n, idx1, idx2)
    pc_1_ngb_n = pc_2_n[idx2]
    pc_2_ngb_n = pc_1_n[idx1]
    # D2 may not exactly match mpeg-pcc-dmetric because of variations in nearest neighbors chosen when at equal distances
    point_to_plane_mse_1 = np.sum(np.sum((pc_1 - pc_1_ngb) * pc_1_ngb_n, axis=1) ** 2) / pc_1.shape[0]
    point_to_plane_mse_2 = np.sum(np.sum((pc_2 - pc_2_ngb) * pc_2_ngb_n, axis=1) ** 2) / pc_2.shape[0]
    point_to_plane_psnr_1 = psnr(point_to_plane_mse_1, max_energy)
    point_to_plane_psnr_2 = psnr(point_to_plane_mse_2, max_energy)
    point_to_plane_result = {
        'psnr_1': point_to_plane_psnr_1,
        'psnr_2': point_to_plane_psnr_2,
        'mse_1': point_to_plane_mse_1,
        'mse_2': point_to_plane_mse_2,
        'psnr_mean': (point_to_plane_psnr_1 + point_to_plane_psnr_2) / 2,
        'mse_mean': (point_to_plane_mse_1 + point_to_plane_mse_2) / 2,
    }
    return point_to_point_result, point_to_plane_result


def find_and_sort_bin_files(scan_dir, sequences):
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
