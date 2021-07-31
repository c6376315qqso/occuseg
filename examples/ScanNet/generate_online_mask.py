import os
import torch 
import numpy as np
import glob

FAR_PLANE = 3
NEAR_PLANE = 0.1

def work(name):
    data3d = torch.load(os.path.join(name, name + '_instance.pth'))
    data3d = [torch.from_numpy(data) for data in data3d]
    pose_files = glob.glob(os.path.join(name, 'pose', '*.txt'))
    
        pose_file = open(scannet_online_list[tbl[0]], "r")
        pose = torch.empty(4,4)
        for i in range(4):
            pose[i] = torch.tensor(list(map(lambda x:float(x), pose_file.readline().split(' '))))
        intrinsic = torch.tensor([[577.590698, 0, 318.905426],
                                [0, 578.729797, 242.683609],
                                [0,0,1]])
        R = pose[:3,:3]
        t = pose[:3, 3]
        points_in_camera = torch.matmul(intrinsic, torch.matmul(R.t(), data3d[0].t() - t.view(-1, 1))).t()
        inside_mask = (points_in_camera[:,2] < FAR_PLANE) & (points_in_camera[:,2] > NEAR_PLANE)
        points_in_camera = points_in_camera / points_in_camera[:,2].view(-1, 1)
        # print(points_in_camera[inside_mask])
        inside_mask = inside_mask & (points_in_camera[:,0] < 640) & (points_in_camera[:,0] >= 0) & (points_in_camera[:,1] < 480) & (points_in_camera[:,1] >= 0)
    return inside_mask