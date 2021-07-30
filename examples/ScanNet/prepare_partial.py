import open3d as o3d
import glob, plyfile, numpy as np, multiprocessing as mp, torch
import copy
import numpy as np
import json
import pdb
import os


#CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
CLASS_LABELS = ['wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']
VALID_CLASS_IDS = np.array([1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
NYUID_TO_LABEL = {}
LABEL_TO_NYUID = {}
NYUID_TO_SEGID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_NYUID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    NYUID_TO_LABEL [VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    NYUID_TO_SEGID[VALID_CLASS_IDS[i]] = i

SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

LABEL_ID_TO_CLASS_ID = {
#    9: 3 #force 'desk' to be the same class_id as 'table'
}
for i, label_id in enumerate(SELECTED_LABEL_IDS):
    if label_id not in LABEL_ID_TO_CLASS_ID:
        LABEL_ID_TO_CLASS_ID[label_id] = i

UNKNOWN_ID = -100
MIN_INSTANCE_SIZE = 600


remapper=np.ones(500)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i
## read coordinates, color, semantic_labels, indices and dists of 2 nearest points

def f(fn): 
    # output_file = output_path + "/" + fn.rsplit("/", 1)[1][:-18]+'.pth'
    # if os.path.exists(output_file)==True:
    #     print("exist:",output_file)
    #     return 0
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3])
    colors=np.ascontiguousarray(v[:,3:6])/255.0 - 0.5          

    position=np.ascontiguousarray(v[:,8:10])
    dis=np.ascontiguousarray(v[:,10:12])

    # filter out very small segements and reassign  instance labels
    
    w = np.zeros((len(coords),2),  dtype = np.int32)
    w[:,:] = UNKNOWN_ID

    semantic_labels_ori = np.array(a.elements[0]['label'])
    instance_labels = np.array(a.elements[0]['instance_label'])

    semantic_labels = np.array(list(
        map(lambda label_id: LABEL_ID_TO_CLASS_ID[label_id] if label_id in LABEL_ID_TO_CLASS_ID else UNKNOWN_ID,
            semantic_labels_ori)))

    for id in range(instance_labels.max()+1):
        instance_indices = (instance_labels == id)
        instance_size = instance_indices.sum()
        # print(instance_size)
        if instance_size > MIN_INSTANCE_SIZE:
            w[instance_indices,0] = semantic_labels[instance_indices]
            w[instance_indices,1] = id

    # print(np.unique(w[:,0]))
    # print(np.unique(w[:,1]))

    # w[:,0] = remapper[]
    # w[:,1] = remapper[]

    json_file = open(fn[:-3]+'0.010000.segs.json')
    region = np.asarray(json.load(json_file)['segIndices'])
    all={
        "coords":coords,
        "colors":colors,
        "w":w,
        'region': region
    }
    # print("save to "+ output_file)
    # # pdb.set_trace()
    # torch.save((coords,colors,w1,w2,position,dis),output_file )
    fileName = fn[:-4] + '_instance.pth'
    torch.save(all, fileName)
    print(fileName)

print("avilable cpus: ", mp.cpu_count())

files = sorted(glob.glob('/media/hdd/zhengtian/Occuseg/data/scannet_partial/instance/partial_1/train/*.ply'))
# print(files[0])
# f(files[0])
p = mp.Pool(processes=mp.cpu_count() - 4)
p.map(f, files)
p.close()
p.join()

files = sorted(glob.glob('/media/hdd/zhengtian/Occuseg/data/scannet_partial/instance/partial_1/val/*.ply'))
p = mp.Pool(processes=mp.cpu_count() - 4)
p.map(f, files)
p.close()
p.join()

files = sorted(glob.glob('/media/hdd/zhengtian/Occuseg/data/scannet_partial/instance/partial_2/train/*.ply'))
p = mp.Pool(processes=mp.cpu_count() - 4)
p.map(f, files)
p.close()
p.join()

files = sorted(glob.glob('/media/hdd/zhengtian/Occuseg/data/scannet_partial/instance/partial_2/val/*.ply'))
p = mp.Pool(processes=mp.cpu_count() - 4)
p.map(f, files)
p.close()
p.join()
# # parallel
# p = mp.Pool(processes=mp.cpu_count())
# p.map(f_gene_ply,files)
# p.close()
# p.join()
 



