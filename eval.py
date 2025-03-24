import argparse
import csv
import os
from datetime import datetime

import numpy as np
import octree_module
import torch

from config import config
from model.kitti_api import KittiAPI
from utils.metrics import get_PSNR
from utils.octree import Octree2Points

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, required=True, help="CFE数据集路径")
parser.add_argument('--decode', action="store_true", help="是否需要解码")
args = parser.parse_args()

TEST_SEQ = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
LEVEL = [8, 9, 10, 11, 12]


# 定义写入CSV文件的函数
def write_to_csv(point_file, bpp, d1psnr, d2psnr, file_path):
    # 检查文件是否存在，如果不存在，则写入标题
    try:
        with open(file_path, mode='r') as file:
            pass
    except FileNotFoundError:
        # 如果文件不存在，写入标题行
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['File', 'BPP', 'D1-PSNR', 'D2-PSNR'])  # 写入标题行

    # 记录每次测试的结果
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([point_file, bpp, d1psnr, d2psnr])  # 将每次测试的结果写入文件


if __name__ == "__main__":
    need_decode = True if args.decode else False
    cfe_path = args.path
    if not os.path.isdir(cfe_path):
        RuntimeError("指定的路径错误")
    levels = LEVEL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 加载一下相关的参数 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    min_bound = np.array(config.kitti.min_bound)
    max_bound = np.array(config.kitti.max_bound)
    coder = None

    for level in levels:
        model_path = 'save/20250310.pth.tar' if level > 12 else 'save/20250207.pth.tar'
        coder = KittiAPI(model_path, static_prob_file="save/kitti_padding_prob.npz", depth=level)
        for seq in TEST_SEQ:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 获取目录下所有文件 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            directory = os.path.join(cfe_path, seq, 'velodyne')
            directory = os.path.expanduser(directory)
            files = os.listdir(directory)
            # 过滤出以.ply结尾的文件
            bin_files = [f for f in files if f.endswith('.bin')]
            # 按文件名升序排序
            bin_files_sorted = sorted(bin_files, key=lambda x: int(x.split('.')[0]))
            # 获取文件的完整路径
            bin_files_path = [os.path.join(directory, f) for f in bin_files_sorted]
            bin_files_path = bin_files_path[:30]

            trans = Octree2Points(min_bound=min_bound, max_bound=max_bound, depth=level)
            octree_converter = octree_module.Octree2Points(min_bound, max_bound, level)
            # 获取当前日期
            current_date = datetime.now().date()
            static_file = f'evaluation/{seq}-{level}.csv'

            for bin_file_path in bin_files_path:
                byte_stream, node_num, start_ptr, org_points = coder.encode(bin_file_path)
                #      ┌──────────────────────────────────────────────────────────┐
                #      │                这里获得的byte_stream即是比特流               │
                #      └──────────────────────────────────────────────────────────┘
                if byte_stream is not None:
                    if need_decode:
                        octree_data, ancestors_data = coder.decode(byte_stream, node_num, start_ptr)
                        points = trans.convert(octree_data)
                    binsz = len(byte_stream) * 8
                    bpp = binsz / org_points.shape[0]  # 换回量化前
                    print(f"BPP: {bpp}")
                    psnr1 = 0
                    psnr2 = 0
                    if need_decode:
                        point_cloud1 = torch.tensor(org_points, dtype=torch.double, device="cuda").unsqueeze(dim=0)
                        point_cloud2 = torch.tensor(points, dtype=torch.double, device="cuda").unsqueeze(dim=0)
                        psnr_error = get_PSNR(org_points, points, test_d2=True)
                        psnr1 = psnr_error['mseF,PSNR (p2point)']
                        psnr2 = psnr_error['mseF,PSNR (p2plane)']
                        print(f"BPP: {bpp} | D1-PSNR: {psnr1} | D2-PSNR: {psnr2}")
                    write_to_csv(bin_file_path, bpp, psnr1, psnr2, static_file)
            coder.encoder_buffer = None  # 清空序列的参考帧缓存
