import os
from numpy.lib.twodim_base import mask_indices
import torch 
import numpy as np
import glob

FAR_PLANE = 3
NEAR_PLANE = 0.1

def work(name):
    if not os.path.exists(os.path.join(name, 'online_masks')):
        os.mkdir(os.path.join(name, 'online_masks'))
    data3d = torch.load(os.path.join(name, name + '_instance.pth'))
    data3d = data3d['coords']
    pose_files = glob.glob(os.path.join(name, 'pose', '*.txt'))
    pose_files.sort(key=lambda x: int(x[x.rfind('/')+1:-4]))
    all_mask = torch.zeros(data3d[0].shape[0]).byte()
    
    for idx, pose_file in enumerate(pose_files):
        if idx % 10 != 0:
            continue
        pose = np.loadtxt(pose_files)
        pose = torch.from_numpy(pose)
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
        masks = []
        all_mask = all_mask | inside_mask
        if torch.sum(all_mask, dim=0) >= all_mask.shape[0] * 0.25:
            mask1 = all_mask.clone()
            print('mask1: ', torch.sum(all_mask, dim=0) / all_mask.shape[0])
        elif torch.sum(all_mask, dim=0) >= all_mask.shape[0] * 0.5: 
            mask2 = all_mask.clone()
            print('mask2: ', torch.sum(all_mask, dim=0) / all_mask.shape[0])
        elif torch.sum(all_mask, dim=0) >= all_mask.shape[0] * 0.75:
            mask3 = all_mask.clone()
            print('mask3: ', torch.sum(all_mask, dim=0) / all_mask.shape[0])
        mask = torch.stack([mask1, mask2, mask3], dim=0)
        torch.save(mask, os.path.join(name, 'm25_50_75.pth'), _use_new_zipfile_serialization=False)
        print(name)


work('scene0000_00')



