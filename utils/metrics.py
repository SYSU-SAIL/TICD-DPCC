import os
import subprocess
import time
import open3d as o3d
import numpy as np
from numpy import ndarray

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue
    return number


def pc_error(infile1, infile2, resolution, normal=False, show=False):
    start_time = time.time()
    # headersF = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
    #            "h.       1(p2point)", "h.,PSNR  1(p2point)",
    #            "mse2      (p2point)", "mse2,PSNR (p2point)", 
    #            "h.       2(p2point)", "h.,PSNR  2(p2point)" ,
    #            "mseF      (p2point)", "mseF,PSNR (p2point)", 
    #            "h.        (p2point)", "h.,PSNR   (p2point)" ]
    # headersF_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
    #                   "mse2      (p2plane)", "mse2,PSNR (p2plane)",
    #                   "mseF      (p2plane)", "mseF,PSNR (p2plane)"]             
    headers = ["mseF      (p2point)", "mseF,PSNR (p2point)"]
    command = str(current_dir + '/pc_error_d' +
                  ' -a ' + infile1 +
                  ' -b ' + infile2 +
                  ' --hausdorff=1 ' +
                  ' --resolution=' + str(resolution))
    if normal:
        headers += ["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command = str(command + ' -n ' + infile1)
    results = {}
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')  # python3.
        if show: print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c = subp.stdout.readline()
    return results


def chamfer_dist(a, b):
    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(a.astype('float32'))
    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(b.astype('float32'))
    distA = pcdA.compute_point_cloud_distance(pcdB)
    distB = pcdB.compute_point_cloud_distance(pcdA)
    distA = np.array(distA) ** 2
    distB = np.array(distB) ** 2
    return distA, distB


def save_ply(points, file_path, normal=False):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if normal:
        point_cloud.estimate_normals()
    # 保存为 ply 格式
    o3d.io.write_point_cloud(file_path, point_cloud, write_ascii=True)
    f = open(file_path)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    if normal:
        lines[7] = 'property float nx\n'
        lines[8] = 'property float ny\n'
        lines[9] = 'property float nz\n'
    fo = open(file_path, "w")
    fo.writelines(lines)


def get_PSNR_attn(f1: ndarray, f2: ndarray, test_d2=False):
    """pc0: origin data;    pc1: decded data
    """
    points1 = f1.astype(np.float32)
    points2 = f2.astype(np.float32)
    centroid = points1.mean(axis=0)
    points1 -= centroid
    points2 -= centroid
    max_value = np.max(np.abs(points1))
    points1 /= max_value
    points2 /= max_value
    time_stamp = int(round(time.time() * 1000))
    outfile1 = os.path.join(current_dir, f"{time_stamp}_original.ply")
    outfile2 = os.path.join(current_dir, f"{time_stamp}_decded.ply")
    save_ply(points1, outfile1, normal=test_d2)
    save_ply(points2, outfile2, normal=test_d2)
    results = pc_error(outfile1, outfile2, resolution=1, normal=test_d2)
    os.remove(outfile1)
    os.remove(outfile2)
    return results


def get_PSNR(f1: ndarray, f2: ndarray, test_d2=False):
    """pc0: origin data;    pc1: decded data
    """
    points1 = f1.astype(np.float32)
    points2 = f2.astype(np.float32)
    points1 = np.round(points1*1000)
    points2 = np.round(points2*1000)
    time_stamp = int(round(time.time() * 1000))
    outfile1 = os.path.join(current_dir, f"{time_stamp}_original.ply")
    outfile2 = os.path.join(current_dir, f"{time_stamp}_decded.ply")
    save_ply(points1, outfile1, normal=test_d2)
    save_ply(points2, outfile2, normal=test_d2)
    results = pc_error(outfile1, outfile2, resolution=30000, normal=test_d2)
    os.remove(outfile1)
    os.remove(outfile2)
    return results
