import sys
# uncomment
# import build.example as example
import torch
# uncomment
# from pytorch3d.ops import knn_points 
import os
import time
import argparse
import collections
import timeit
import scipy.spatial as sp
from sacred import Experiment
from sacred.observers import FileStorageObserver
ex = Experiment("test_time1")
ex.observers.append(FileStorageObserver.create('./test_time_log'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
if torch.cuda.is_available():
    torch.cuda.init()
@ex.config         
def cfg():
    input_dir="~/Data/Kinect/Synthetic/train/original_xyz"
    output_dir="~/Data/Kinect/Synthetic/train/original_xyz_patches"
    use_density=0 
    use_knn=0 
    use_ballquery=1 
    patch_size=256
    radius = 0.2

    num_iter = 100
    problemfilelist = ["dragon.xyz"]

def normalized(points): # normalize the points to [0, 1]
    max = np.max(points, axis=0)
    min = np.min(points, axis=0)
    points -= min
    points /= (max - min)
    return points
# example.help()
# help(example)
def find_duplicate(arr):
    counter=collections.Counter(arr)
    # print(counter)
    # return a dictionary , key is the element in arr, value is the number of the element
    dic = {key:val for key, val in counter.items() if val > 1}
    return dic
@ex.capture
def test_time(input_dir, output_dir, use_knn, use_density, use_ballquery, patch_size, radius, num_iter, problemfilelist): 


    all_time = []
    print(f" len(input_dir) = {len(os.listdir(input_dir))}")
    for idx,filename in enumerate(sorted(os.listdir(input_dir))):
        if filename  in problemfilelist:
            continue
        if filename.endswith(".xyz"):
            print(idx,filename)
            input_file = os.path.join(input_dir, filename)
            points = np.loadtxt(input_file)
            normalized(points)
            kdtree = None
            if use_ballquery:
                def build_kdtree():
                    global kdtree
                    kdtree = sp.cKDTree(points)
                kd_time = timeit.timeit(build_kdtree, number=1) 
                print(f"kd_time = {kd_time}")
            print(f"N={points.shape[0]} ")
            for startidx in range(1, 302, 5):
                for max_points in range(patch_size, patch_size+1, 1):
                    # print("start idx = ", startidx)
                    output_file = os.path.join(output_dir, filename.replace(".xyz", f"nums{max_points}_idx{startidx}.xyz"))
                    # np.savetxt(os.path.join(output_dir,filename), points, fmt='%.6f')

                    startpoint = torch.tensor(points[startidx]).unsqueeze(0).unsqueeze(0).float()
                    N = points.shape[0]
                    if use_density:
                        def code_to_test():
                            global radius
                            if N <= 2000:
                                radius = 0.2
                            elif N <= 4000:
                                radius = 0.15
                            elif N <= 8000:
                                radius = 0.1
                            elif N <= 16000:
                                radius = 0.06
                            elif N <= 20000:
                                radius = 0.04
                            elif N<=30000:
                                radius = 0.03  
                            else:
                                radius = 0.06
                            idx = example.process_pointsV1(torch.from_numpy(points).float().contiguous().to('cuda'), startidx, radius, max_points, num_iter)
                            # idx = idx.cpu().numpy()
                            # selected_points = points[idx]
                            # np.savetxt(output_file, selected_points, fmt='%.6f')
                        idx = example.process_pointsV1(torch.from_numpy(points).float().contiguous().to('cuda'), startidx, radius, max_points, num_iter)
                        # print(f"radius = {radius}")
                        idx = idx.cpu().numpy()
                        selected_points = points[idx]
                        np.savetxt(output_file, selected_points, fmt='%.6f')
                        temp_time = timeit.timeit(code_to_test, number=1)
                    if use_knn:
                        def code_to_test():
                            _, idx, select_points_knn = knn_points(startpoint, torch.from_numpy(points).unsqueeze(0).float(), K=max_points, return_nn=True)
                            # idx = idx.squeeze().numpy()                       
                        temp_time = timeit.timeit(code_to_test, number=1)
                        _, idx, select_points_knn = knn_points(startpoint, torch.from_numpy(points).unsqueeze(0).float(), K=max_points, return_nn=True)
                        idx = idx.squeeze().numpy()
                    if use_ballquery:
                        dic_r = {
                            256:0.2,
                            512:0.3,
                            1024:0.5
                            }
                        radius = dic_r[max_points]  # 0.5 0.3 0.2

                        def patch_sampling( patch_pts):
                            points_per_patch = max_points
                            if patch_pts.shape[0] > points_per_patch:
                            
                                sample_index = np.random.choice(range(patch_pts.shape[0]), points_per_patch, replace=False) #从patch_pts.shape[0]中取得points_per_patch个点。不放回取
                            else:
                                sample_index = np.random.choice(range(patch_pts.shape[0]), points_per_patch) #放回取

                            return sample_index
                        def code_to_test():
                            global kdtree,radius
                            # kdtree = sp.cKDTree(points)

                            idx = kdtree.query_ball_point(points[startidx], radius)
                            idx = np.array(idx)
                            selected_points = points[idx]
                            new_idx = patch_sampling(selected_points)
                        temp_time = timeit.timeit(code_to_test, number=1)
                        bbdiag = float(np.linalg.norm(points.max(0) - points.min(0), 2))
                        radius=(bbdiag * radius)
                        idx = kdtree.query_ball_point(points[startidx], radius)
                        idx = np.array(idx)
                        selected_points = points[idx]
                        idx = patch_sampling(selected_points)
                        selected_points = selected_points[idx]                    
                        np.savetxt(output_file, selected_points, fmt='%.6f')

                    duplicate_list = find_duplicate(idx)
                    if(duplicate_list):
                        # print(duplicate_list)
                        print(f"idx has duplicate points")    
                        # raise ValueError("idx has duplicate points")
                    all_time.append(temp_time)
            if use_ballquery:
                all_time[-1]+=kd_time

    # print(f"all time ={all_time}")
    # all_time = all_time[1:]
    print(f"len={len(all_time)},average time = {sum(all_time)/len(all_time):.6f}")
@ex.capture
def prepare_data(input_dir, output_dir, use_density):
    if not os.path.exists(input_dir):
        raise ValueError("dir not exist")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("dir already exist")
    if use_density:
        print(f"1+2=",example.add(1, 2))    
@ex.automain
def main():
    prepare_data()
    test_time()              
