# import open3d



from examples.ScanNet.discriminative import ConsistencyLoss_p2i,ConsistencyLoss_i2i, Consistency_Evaler,ConsistencyLoss_p2p, Embedding_Evaler
from functools import partial
from examples.ScanNet.datasets.scannet import ScanNetOnline
# import open3d
from datasets import ScanNet
from utils import evaluate_scannet, evaluate_stanford3D,WeightedCrossEntropyLoss, FocalLoss, label2color,evaluate_single_scan,cost2color
from model import ThreeVoxelKernel
from model import DenseUNet,InstanceDenseUNet,LearningBWDenseUNet,ClusterSegNet, UncertainDenseUNet
from config import get_args
from config import ArgsToConfig
from scipy.io import savemat
import pdb
import sys
sys.path.insert(0, '../../../jsis3d/')

from lovasz_losses import lovasz_hinge,lovasz_softmax,lovasz_hinge_flat
from sklearn.cluster import MeanShift
import scipy.stats as stats
from model import *
from discriminative import DiscriminativeLoss,ClassificationLoss,DriftLoss
from torch_scatter import scatter_max,scatter_mean,scatter_std,scatter_sub,scatter_min,scatter_add,scatter_div
#from knn_cuda import KNN
import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import nvgpu

import sys, os, time
import logging
import math
import json
from tqdm import tqdm
import os
import bcelosses

# only holds when the optimization is zero! a bad cost function, could be further optimized?
# cannot explain well, how should we encourage the displacements to be correct?
#A larger v generalizes better
DISCRIMINATIVE_DELTA_V = 0.2
DISCRIMINATIVE_DELTA_D = 1.5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training logger')
logger.setLevel(logging.DEBUG)
Model = ThreeVoxelKernel


def visualize_point_cloud(batch,predictions):
    tbl = batch['id']
    for k,idx in enumerate(tbl):
        index = batch['x'][0][:,3] == k
        pos = batch['x'][0][index,0:3].data.cpu().numpy()
        color = batch['x'][1][index,0:3].data.cpu().numpy() + 0.5
        label = batch['y'][index,0].data.cpu().numpy()
        if(label.shape[0] > 0):
            predicted_label = predictions[index,:].max(1)[1]
            prob = predictions[index,:]
            cost_color = cost2color(prob,label)
            predicted_colors = label2color(predicted_label)
            label_colors = label2color(label)
            ref = torch.from_numpy(pos).float().cuda()
            pcd_ori = open3d.geometry.PointCloud()
            pcd_ori.points = open3d.Vector3dVector(pos)
            pcd_ori.colors = open3d.Vector3dVector(cost_color)
            pcd_gt_label = open3d.geometry.PointCloud()
            gt_pos = pos
            gt_pos[:,0] = pos[:,0] + 400
            pcd_gt_label.points = open3d.Vector3dVector(gt_pos)
            pcd_gt_label.colors = open3d.Vector3dVector(label_colors)
            pcd_predict_label = open3d.geometry.PointCloud()
            predict_pos = gt_pos
            predict_pos[:,0] = predict_pos[:,0] + 400
            pcd_predict_label.points = open3d.Vector3dVector(predict_pos)
            pcd_predict_label.colors = open3d.Vector3dVector(predicted_colors)
            current_iou = iou_evaluate(predicted_label.numpy(), label, train_writer, 0,class_num = config['class_num'] ,  topic='valid')
            #predictions
            if config['dataset'] == 'stanford3d':
                fileName = batch['pth_file'][k][0:batch['pth_file'][k].rfind('/')] + '_' + "%.2f" % current_iou
            elif config['dataset'] == 'scannet':
                fileName = batch['pth_file'][k][0:-4] + '_' + "%.2f" % current_iou
            fori = fileName + '_ori.pcd'
            fgt = fileName + '_gt.pcd'
            fpredict = fileName + '_predict.pcd'
            open3d.write_point_cloud(fori, pcd_ori)
            open3d.write_point_cloud(fgt, pcd_gt_label)
            open3d.write_point_cloud(fpredict, pcd_predict_label)


volumetric_sizes_per_category = np.loadtxt("sizes.txt")



def evaluate_instance(net, config, global_iter):
    valOffsets = config['valOffsets']
    val_data_loader = config['val_data_loader']
    valLabels = config['valLabels']
    pdict = {'semantics': [], 'instances': []}
    total = np.zeros(config['class_num'])
    fps   = [[] for i in range(config['class_num'])]
    tps   = [[] for i in range(config['class_num'])]
    criterion = {}
    criterion['discriminative'] = DiscriminativeLoss(
        DISCRIMINATIVE_DELTA_D,
        DISCRIMINATIVE_DELTA_V
    )
    criterion['nll'] = nn.functional.cross_entropy
    criterion['regression'] = nn.L1Loss()
    with torch.no_grad():
        net.eval()
        store = torch.zeros(valOffsets[-1], config['class_num'])
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        time_val_start = time.time()
        for rep in range(1, 2):
            locs = None
            pth_files = []
            for i, batch in enumerate(val_data_loader):
                batch['x'][1] = batch['x'][1].cuda()
                torch.cuda.empty_cache()
                predictions, feature, embeddings, offsets, displacements, bw, occupancy = net(batch['x'])
                batch['y'] = batch['y'].cuda()
                batch['region_masks'] =  batch['region_masks'].cuda()
                batch['offsets'] =  batch['offsets'].cuda()
                batch['displacements'] =  batch['displacements'].cuda()
                calculate_cost(predictions, embeddings, offsets, displacements, bw, criterion, batch, occupancy)
#                print("computing semantic embedding test: ", i)
                semantics = np.argmax(predictions.cpu().numpy(), axis=-1)
                instances = []
                tbl = batch['id']


                predictions = predictions.cpu()
                store.index_add_(0, batch['point_ids'], predictions)
                averaged_predictions = store[batch['point_ids'],:] / rep
                if(rep == 1):
                    for k,idx in enumerate(tbl):
                        index = (batch['x'][0][:,config['dimension']] == k)
                        embedding = embeddings[index,:].cpu().numpy()
                        semantic = averaged_predictions.max(1)[1][index].cpu().numpy()

                        size = batch['sizes'][k].item()
    #                    y = MeanShift(1.5, n_jobs=8).fit_predict(embedding)
    #                    instances.append(y)
                        offline_data = {}
                        offline_data['xyz'] = batch['x'][0][index,:config['dimension']].cpu().numpy()
                        offline_data['feature'] = batch['x'][1][index,:].cpu().numpy()
                        offline_data['occupancy'] = occupancy[index,0].cpu().numpy()
                        offline_data['true_occupancy'] = batch['instance_sizes'][index].cpu().numpy()
                        offline_data['pred_semantic'] = semantic
                        offline_data['pred_semantic_probability'] = averaged_predictions[index,:].cpu().numpy()
                        offline_data['pred_embedding'] = embedding
                        offline_data['regions'] = batch['regions'][index].cpu().numpy()
                        offline_data['true_semantic'] = batch['y'][index,0].cpu().numpy()
                        offline_data['true_instance'] = batch['y'][index,1].cpu().numpy()
                        offline_data['pred_offsets'] = offsets[index].cpu().numpy()
                        offline_data['true_offsets'] = batch['offsets'][index].cpu().numpy()
                        offline_data['pred_displacements'] = displacements[index].cpu().numpy()
                        offline_data['true_displacements'] = batch['displacements'][index].cpu().numpy()
                        offline_data['pred_bw'] = bw[index,:].cpu().numpy()
                        offline_data['scale'] = config['scale']
                        filename = batch['pth_file'][k]
                        base_path, file_path = os.path.split(filename)

                        if not os.path.exists(os.path.join(base_path, args.taskname)):
                            os.makedirs(os.path.join(base_path, args.taskname))
                        filename = os.path.join(base_path, args.taskname, file_path[:-3] + 'npz')
#                        np.savez(filename , **offline_data)
                        print('save: ', filename)

            print('infer', rep, 'Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count / len(dataset.val) / 1e6,
                        'MegaHidden', scn.forward_pass_hidden_states / len(dataset.val) / 1e6, 'time=',
                        time.time() - time_val_start,
                        's')
            predLabels = store.max(1)[1].numpy()
            topic = 'valid_' + str(rep)
            iou_evaluate(predLabels, valLabels, train_writer, global_iter, class_num = config['class_num'] , topic=topic)
            train_writer.add_scalar("valid/MegaMulAdd", scn.forward_pass_multiplyAdd_count / len(dataset.val) / 1e6,
                                    global_step=global_iter)
            train_writer.add_scalar("valid/MegaHidden", scn.forward_pass_hidden_states / len(dataset.val) / 1e6,
                                    global_step=global_iter)


            if config['evaluate']:
                savemat('pred.mat', {'p':store.numpy(),'val_offsets':np.hstack(valOffsets), 'v': valLabels})

        predLabels = store.max(1)[1].numpy()
        iou_evaluate(predLabels, valLabels, train_writer, global_iter,class_num = config['class_num'] ,  topic='valid')
        train_writer.add_scalar("valid/time", time.time() - time_val_start, global_step=global_iter)


def calculate_cost(predictions, embeddings, offsets, displacements, bw, criterion, batch, occupancy):
    tbl = batch['id']
    #lovasz_softmax(predictions, batch['y'][:,0], ignore = -100)
    SemanticLoss = criterion['nll'](predictions, batch['y'][:,0])
    EmbeddingLoss = torch.zeros(1,dtype=torch.float32).cuda()
    OccupancyLoss = torch.zeros(1,dtype=torch.float32).cuda()
    DisplacementLoss = torch.zeros(1,dtype=torch.float32).cuda()
    loss_classification = torch.zeros(1,dtype=torch.float32).cuda()
    loss_drift = torch.zeros(1,dtype=torch.float32).cuda()
    instance_iou = torch.zeros(1,dtype=torch.float32).cuda()

    batchSize = 0
    forground_indices = batch['y'][:,0] > 1
    regressed_pose = batch['x'][0][:,0:3].cuda() / config['scale'] - displacements
    pose = batch['x'][0][:,0:3].cuda() / config['scale']
    displacements_gt = batch['displacements'].cuda()
    occupancy_gt = batch['instance_sizes'].cuda().view(-1,1)
    for count,idx in enumerate(tbl):
        index = (batch['x'][0][:,config['dimension']] == count)
        embedding = embeddings[index,:].view(1,-1,embeddings.shape[1])
        instance_mask = batch['instance_masks'][index].view(1,-1).cuda().type(torch.long)
        pred_semantics = batch['y'][index,0]
        EmbeddingLoss += criterion['discriminative'](embedding, instance_mask)

        displacement_error = torch.zeros(1,dtype=torch.float32).cuda()
        occupancy_error = torch.zeros(1,dtype=torch.float32).cuda()
        cluster_size = 0

        displacement_cluster_error = scatter_mean(torch.norm(displacements[index,:]- displacements_gt[index,:], dim = 1), instance_mask.view(-1),dim = 0)
#        true_occupancy = scatter_mean(torch.norm(occupancy_gt[index,:], dim = 1 ), instance_mask.view(-1),dim = 0)
#        pred_occupancy = scatter_mean(torch.norm(occupancy[index,:], dim = 1 ), instance_mask.view(-1),dim = 0)

        occupancy_cluster_error = scatter_mean(torch.norm(occupancy[index,:] - occupancy_gt[index,:], dim = 1), instance_mask.view(-1),dim = 0)
        occupancy_cluster_std = scatter_std(occupancy[index,:], instance_mask.view(-1),dim = 0)
        mask_size = instance_mask[0,:].max() + 1
        for mid in range(mask_size):
            instance_indices = (instance_mask[0,:]==mid)
            cls = pred_semantics[instance_indices][0]
            if(cls > 1):
                displacement_error += displacement_cluster_error[mid]
                occupancy_error += occupancy_cluster_error[mid] +  occupancy_cluster_std[mid]
#                current_occupancy = occupancy[index,:]
#                median_occupancy = torch.median(current_occupancy[instance_indices,:])
#                print(cls.item(), torch.exp(pred_occupancy[mid]).item(),torch.exp(median_occupancy).item(), torch.exp(true_occupancy[mid]).item())
                cluster_size += 1
        if cluster_size > 0:
            OccupancyLoss += occupancy_error / cluster_size
            DisplacementLoss += displacement_error / cluster_size
        loss_c, i_iou = ClassificationLoss(embedding,bw[index,:].view(1,-1,2), regressed_pose[index,:].view(1,-1,3), pose[index,:].view(1,-1,3), instance_mask,pred_semantics)
        loss_classification += loss_c
        instance_iou += i_iou

#        loss_drift += 50 * DriftLoss(embedding,mask.cuda(),pred_semantics,regressed_pose[index,:].view(1,-1,3),batch['offsets'][index,:], pose[index,:].view(1,-1,3))      #We want to align anchor points to mu as close as possible

        batchSize += 1
    EmbeddingLoss /= batchSize
    OccupancyLoss /= batchSize
    DisplacementLoss /= batchSize
    loss_classification /= batchSize
    loss_drift /= batchSize
    instance_iou /= batchSize
    PreOccupancyLoss = criterion['regression'](occupancy[forground_indices].view(-1), batch['instance_sizes'].cuda()[forground_indices])
    PreDisplacementLoss = criterion['regression'](displacements[forground_indices ,:], batch['displacements'].cuda()[forground_indices ,:]) * config['displacement_weight']
    #print('previous occupancy loss: ', PreOccupancyLoss.item(),OccupancyLoss.item(),'    ', PreDisplacementLoss.item(),DisplacementLoss.item())
    RegressionLoss = criterion['regression'](offsets[forground_indices ], batch['offsets'].cuda()[forground_indices ]) * config['regress_weight']
    #del RegressionLoss
    return {'semantic_loss': SemanticLoss, 'embedding_loss':EmbeddingLoss, 'regression_loss':RegressionLoss, 'displacement_loss':DisplacementLoss,
            'classification_loss':loss_classification, 'drift_loss':loss_drift, 'instance_iou':instance_iou, 'occupancy_loss': OccupancyLoss}

def calculate_cost_online_eval(predictions, embeddings, offsets, displacements, bw, criterion, batch):
    tbl = batch['id']
    #lovasz_softmax(predictions, batch['y'][:,0], ignore = -100)
    SemanticLoss = criterion['nll'](predictions, batch['y'][:,0])
    EmbeddingLoss = torch.zeros(1,dtype=torch.float32).cuda()
    DisplacementLoss = torch.zeros(1,dtype=torch.float32).cuda()
    loss_classification = torch.zeros(1,dtype=torch.float32).cuda()
    loss_drift = torch.zeros(1,dtype=torch.float32).cuda()
    instance_iou = torch.zeros(1,dtype=torch.float32).cuda()

    batchSize = 0
    forground_indices = batch['y'][:,0] > 1
    regressed_pose = batch['x'][0][:,0:3].cuda() / config['scale'] - displacements
    pose = batch['x'][0][:,0:3].cuda() / config['scale']
    displacements_gt = batch['displacements'].cuda()
    for count,idx in enumerate(tbl):
        index = (batch['x'][0][:,config['dimension']] == count)
        embedding = embeddings[index,:].view(1,-1,embeddings.shape[1])
        instance_mask = batch['instance_masks'][index].view(1,-1).cuda().type(torch.long)
        pred_semantics = batch['y'][index,0]
        emb_loss, l_v, l_d, l_r = criterion['discriminative'](embedding, instance_mask)
        EmbeddingLoss += emb_loss
        displacement_error = torch.zeros(1,dtype=torch.float32).cuda()
        cluster_size = 0

        displacement_cluster_error = scatter_mean(torch.norm(displacements[index,:]- displacements_gt[index,:], dim = 1), instance_mask.view(-1),dim = 0)
#        true_occupancy = scatter_mean(torch.norm(occupancy_gt[index,:], dim = 1 ), instance_mask.view(-1),dim = 0)
#        pred_occupancy = scatter_mean(torch.norm(occupancy[index,:], dim = 1 ), instance_mask.view(-1),dim = 0)

        mask_size = instance_mask[0,:].max() + 1
        for mid in range(mask_size):
            instance_indices = (instance_mask[0,:]==mid)
            cls = pred_semantics[instance_indices][0]
            if(cls > 1):
                displacement_error += displacement_cluster_error[mid]
#                current_occupancy = occupancy[index,:]
#                median_occupancy = torch.median(current_occupancy[instance_indices,:])
#                print(cls.item(), torch.exp(pred_occupancy[mid]).item(),torch.exp(median_occupancy).item(), torch.exp(true_occupancy[mid]).item())
                cluster_size += 1
        if cluster_size > 0:
            DisplacementLoss += displacement_error / cluster_size
        loss_c, i_iou = ClassificationLoss(embedding,bw[index,:].view(1,-1,2), regressed_pose[index,:].view(1,-1,3), pose[index,:].view(1,-1,3), instance_mask,pred_semantics)
        loss_classification += loss_c
        instance_iou += i_iou

#        loss_drift += 50 * DriftLoss(embedding,mask.cuda(),pred_semantics,regressed_pose[index,:].view(1,-1,3),batch['offsets'][index,:], pose[index,:].view(1,-1,3))      #We want to align anchor points to mu as close as possible

        batchSize += 1
    EmbeddingLoss /= batchSize
    DisplacementLoss /= batchSize
    loss_classification /= batchSize
    loss_drift /= batchSize
    instance_iou /= batchSize
    RegressionLoss = criterion['regression'](offsets[forground_indices ], batch['offsets'].cuda()[forground_indices ]) * config['regress_weight']
    #del RegressionLoss
    return {'semantic_loss': SemanticLoss, 'embedding_loss':EmbeddingLoss, 'regression_loss':RegressionLoss, 'displacement_loss':DisplacementLoss,
            'classification_loss':loss_classification, 'drift_loss':loss_drift, 'instance_iou':instance_iou}

def calculate_cost_online(predictions, embeddings, offsets, displacements, bw, criterion, batch, uncertain, epoch, config):
    tbl = batch['id']
    #lovasz_softmax(predictions, batch['y'][:,0], ignore = -100)
    SemanticLoss = criterion['nll'](predictions, batch['y'][:,0])
    EmbeddingLoss = torch.zeros(1,dtype=torch.float32).cuda()
    UncertainLoss = torch.zeros(1,dtype=torch.float32).cuda()
    DisplacementLoss = torch.zeros(1,dtype=torch.float32).cuda()
    loss_classification = torch.zeros(1,dtype=torch.float32).cuda()
    loss_drift = torch.zeros(1,dtype=torch.float32).cuda()
    instance_iou = torch.zeros(1,dtype=torch.float32).cuda()
    loss_consistent = torch.zeros(1,dtype=torch.float32).cuda()

    loss_var = torch.zeros(1,dtype=torch.float32).cuda().detach()
    loss_dis = torch.zeros(1,dtype=torch.float32).cuda().detach()
    loss_reg = torch.zeros(1,dtype=torch.float32).cuda().detach()

    uncertain_num = 0
    uncertain_tp = 0
    uncertain_fp = 0
    uncertain_tn = 0
    uncertain_fn = 0
    tot_num = 0
    consistency_percent = 0
    uncertain_batch_num = 0
    batchSize = 0
    forground_indices = batch['y'][:,0] > 1
    regressed_pose = batch['x'][0][:,0:3].cuda() / config['scale'] - displacements
    pose = batch['x'][0][:,0:3].cuda() / config['scale']
    displacements_gt = batch['displacements'].cuda()
    scene_masks_list = batch['scene_masks_list']
    
    emb_instance_precision = 0
    
    for count,idx in enumerate(tbl):
        scene_masks = scene_masks_list[count]
        complete_batch_id = (count + 1) * batch['num_per_scene'] - 1
        complete_index = (batch['x'][0][:,config['dimension']] == complete_batch_id)

        instance_mask = batch['instance_masks'][complete_index].type(torch.long).view(1,-1).cuda()
        max_instances_id = instance_mask[0, :].max()
        instance_sizes = torch.zeros([batch['num_per_scene'], max_instances_id + 1]).cuda()
        instance_cls = torch.zeros(max_instances_id + 1).cuda()
        torch.cuda.empty_cache()


        indexs = []
        for partial_id in range(batch['num_per_scene']):
            
            batch_id = count * batch['num_per_scene'] + partial_id
            scene_mask = scene_masks[partial_id]
            index = (batch['x'][0][:,config['dimension']] == batch_id)
            indexs += [index]
            # if torch.sum(scene_mask) != scene_mask.shape[0]:
            ##################### uncertain loss
            if epoch >= config['uncertain_st_epoch']:  
                uncertain_gt = torch.argmax(predictions[complete_index][scene_mask], dim=-1) != torch.argmax(predictions[index], -1)
                uncertain_gt = uncertain_gt.detach()
                uncertain_gt = uncertain_gt.view(-1,1).float()
                uncertain_num += torch.sum(uncertain_gt).item()
                tot_num += uncertain_gt.shape[0]
                uncertain_gt_byte = uncertain_gt.byte()
                uncertain_pred_cls = (uncertain[index] > 0.5).detach()
                uncertain_tp += torch.sum(uncertain_gt_byte * uncertain_pred_cls).item()
                uncertain_tn += torch.sum((uncertain_gt_byte==0) & (uncertain_pred_cls==0)).item()
                uncertain_fp += torch.sum((uncertain_gt_byte==0) & (uncertain_pred_cls==1)).item()
                uncertain_fn += torch.sum((uncertain_gt_byte==1) & (uncertain_pred_cls==0)).item()
                
                uncertain_batch_num += 1

                UncertainLoss += criterion['binnary_classification'](uncertain[index], uncertain_gt)
            ####################
            embedding = embeddings[index,:].view(1,-1,embeddings.shape[1])
            instance_mask = batch['instance_masks'][index].view(1,-1).cuda().type(torch.long)
            pred_semantics = batch['y'][index,0]
            emb_loss, l_v, l_d, l_r = criterion['discriminative'](embedding, instance_mask)
            EmbeddingLoss += emb_loss
            loss_var += l_v
            loss_dis += l_d
            loss_reg += l_r
            displacement_error = torch.zeros(1,dtype=torch.float32).cuda()
            cluster_size = 0

            displacement_cluster_error = scatter_mean(torch.norm(displacements[index,:]- displacements_gt[index,:], dim = 1), instance_mask.view(-1),dim = 0)

            mask_size = instance_mask[0,:].max() + 1
            for mid in range(mask_size):
                instance_indices = (instance_mask[0,:]==mid)
                instance_sizes[partial_id, mid] = torch.sum(instance_indices)
                if instance_sizes[partial_id, mid] == 0:
                    continue
                cls = pred_semantics[instance_indices][0]
                instance_cls[mid] = cls
                if(cls > 1):
                    displacement_error += displacement_cluster_error[mid]
                    cluster_size += 1
            if cluster_size > 0:
                DisplacementLoss += displacement_error / cluster_size
            loss_c, i_iou = ClassificationLoss(embedding,bw[index,:].view(1,-1,2), regressed_pose[index,:].view(1,-1,3), pose[index,:].view(1,-1,3), instance_mask, pred_semantics)
            loss_classification += loss_c
            instance_iou += i_iou

            batchSize += 1

        loss_consis, consis_percent = ConsistencyLoss_p2i(embeddings, indexs, batch['instance_masks'].long().cuda(), max_instances_id, instance_sizes, instance_cls, batch['num_per_scene'])
        loss_consistent += loss_consis
        consistency_percent += consis_percent

        emb_instance_precision += Embedding_Evaler(embeddings, indexs, batch['instance_masks'].long().cuda(), max_instances_id, instance_sizes, instance_cls, batch['num_per_scene'], pose)
                
        

    if uncertain_batch_num > 0:
        UncertainLoss /= uncertain_batch_num
    UncertainLoss *= config['uncertain_task_weight']
    loss_consistent /= len(tbl)
    loss_consistent *= config['consistency_weight']
    consistency_percent /= len(tbl)
    emb_instance_precision /= len(tbl)
    
    EmbeddingLoss /= batchSize
    loss_var /= batchSize
    loss_dis /= batchSize
    loss_reg /= batchSize
    DisplacementLoss /= batchSize
    loss_classification /= batchSize
    loss_drift /= batchSize
    instance_iou /= batchSize
    
    
    if torch.sum(forground_indices) != 0:
        RegressionLoss = criterion['regression'](offsets[forground_indices], batch['offsets'].cuda()[forground_indices]) * config['regress_weight']
    else:
        RegressionLoss = torch.zeros(1, dtype=torch.float32).cuda()


    return {'semantic_loss': SemanticLoss, 'embedding_loss':EmbeddingLoss, 'regression_loss':RegressionLoss, 'displacement_loss':DisplacementLoss,
            'classification_loss':loss_classification, 'drift_loss':loss_drift, 'instance_iou':instance_iou, 'uncertain_loss': UncertainLoss, 'uncertain_num': uncertain_num,
            'uncertain_tp':uncertain_tp, 'uncertain_tn':uncertain_tn, 'uncertain_fp':uncertain_fp, 'uncertain_fn':uncertain_fn, 'tot_num':tot_num, 'consistent_loss':loss_consistent,
            'consistency_percent': consistency_percent, 'emb_instance_precision': emb_instance_precision, 'emb_var':loss_var, 'emb_dis': loss_dis, 'emb_reg':loss_reg}


def evaluate(net, config, global_iter):
    valOffsets = config['valOffsets']
    val_data_loader = config['val_data_loader']
    valLabels = config['valLabels']


    criterion = {}
    criterion['discriminative'] = DiscriminativeLoss(
        DISCRIMINATIVE_DELTA_D,
        DISCRIMINATIVE_DELTA_V
    )
    criterion['nll'] = nn.functional.cross_entropy
    criterion['regression'] = nn.L1Loss()
    with torch.no_grad():
        net.eval()
        store = torch.zeros(valOffsets[-1], config['class_num'])
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        time_val_start = time.time()
        for rep in range(1, 1 + dataset.val_reps):
            locs = None
            pth_files = []
            epoch_len = len(config['val_data_loader'])
            regression_loss = 0
            semantic_loss = 0
            embedding_loss = 0
            displacement_loss = 0
            drift_loss = 0
            classification_loss = 0
            occupancy_loss = 0
            instance_iou = 0
            for i, batch in enumerate(val_data_loader):
                torch.cuda.empty_cache()
#                print("before net 0", nvgpu.gpu_info()[0]['mem_used'])
                batch['x'][1] = batch['x'][1].cuda()
                predictions, feature, embeddings, offsets, displacements, bw, occupancy = net(batch['x'])
#                print("after net 0", nvgpu.gpu_info()[0]['mem_used'])

                batch['y'] = batch['y'].cuda()
                batch['region_masks'] =  batch['region_masks'].cuda()
                batch['offsets'] =  batch['offsets'].cuda()
                batch['displacements'] =  batch['displacements'].cuda()
                batch_size = 0
                tbl = batch['id']

                losses = calculate_cost(predictions,embeddings,offsets,displacements,bw, criterion,batch, occupancy)
                regression_loss += losses['regression_loss'].item()
                embedding_loss += losses['embedding_loss'].item()
                semantic_loss += losses['semantic_loss'].item()
                displacement_loss += losses['displacement_loss'].item()
                classification_loss += losses['classification_loss'].item()
                occupancy_loss += losses['occupancy_loss'].item()
                drift_loss += losses['drift_loss'].item()
                instance_iou += losses['instance_iou'].item()
                predictions = predictions.cpu()
                store.index_add_(0, batch['point_ids'], predictions)
                #visualize based on predictions, use open3D for convinence
                if config['evaluate']:
                    visualize_point_cloud(batch,predictions)
#                            open3d.visualization.draw_geometries([pcd_ori, pcd_gt_label, pcd_predict_label])

                # loop for all val set every snap shot is tooooo slow, pls use val.py to check individually
                # if save_ply:
                #     print("evaluate data: ", i)
                #     iou.visualize_label(batch, predictions, rep, save_dir=config['checkpoints_dir'] +)
            predLabels = store.max(1)[1].numpy()
            topic = 'valid_' + str(rep)
            iou_evaluate(predLabels, valLabels, train_writer, global_iter, class_num = config['class_num'] , topic=topic)
            train_writer.add_scalar(topic + "/epoch_avg_displacement_closs",  displacement_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_regression_closs",  regression_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_embedding_closs", embedding_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_semantic_closs", semantic_loss/ epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_classification_closs", classification_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_occupancy_loss", occupancy_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_drift_closs", drift_loss/ epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_instance_precision", instance_iou/ epoch_len, global_step=global_iter)

            print('infer', rep, 'time=', time.time() - time_val_start, 's')
            if config['evaluate']:
                savemat('pred.mat', {'p':store.numpy(),'val_offsets':np.hstack(valOffsets), 'v': valLabels})
            torch.cuda.empty_cache()

        predLabels = store.max(1)[1].numpy()
        iou_evaluate(predLabels, valLabels, train_writer, global_iter,class_num = config['class_num'] ,  topic='valid')
        train_writer.add_scalar("valid/time", time.time() - time_val_start, global_step=global_iter)

def evaluate_online(net, config, global_iter):
    valOffsets = config['valOffsets']
    val_data_loader = config['val_data_loader']
    valLabels = config['valLabels']


    criterion = {}
    criterion['discriminative'] = DiscriminativeLoss(
        DISCRIMINATIVE_DELTA_D,
        DISCRIMINATIVE_DELTA_V
    )
    criterion['nll'] = nn.functional.cross_entropy
    criterion['regression'] = nn.L1Loss()
    if config['bceloss'] == 'weighted_bce':
        criterion['binnary_classification'] = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['uncertain_weight']]).cuda())
    elif config['bceloss'] == 'focal_loss':
        criterion['binnary_classification'] = bcelosses.BCEFocalLoss(alpha=config['uncertain_weight']/(config['uncertain_weight'] + 1))
    elif config['bceloss'] == 'lovasz':
        criterion['binnary_classification'] = lovasz_hinge

    with torch.no_grad():
        net.eval()
        store = torch.zeros(valOffsets[-1], config['class_num'])
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        time_val_start = time.time()
        for rep in range(1, 1 + dataset.val_reps):
            locs = None
            pth_files = []
            epoch_len = len(config['val_data_loader'])
            regression_loss = 0
            semantic_loss = 0
            embedding_loss = 0
            displacement_loss = 0
            drift_loss = 0
            classification_loss = 0
            uncertain_loss = 0
            instance_iou = 0
            for i, batch in enumerate(val_data_loader):
                torch.cuda.empty_cache()
#                print("before net 0", nvgpu.gpu_info()[0]['mem_used'])
                batch['x'][1] = batch['x'][1].cuda()
                predictions, feature, embeddings, offsets, displacements, bw, uncertain = net(batch['x'])
#                print("after net 0", nvgpu.gpu_info()[0]['mem_used'])

                batch['y'] = batch['y'].cuda()
                batch['region_masks'] =  batch['region_masks'].cuda()
                batch['offsets'] =  batch['offsets'].cuda()
                batch['displacements'] =  batch['displacements'].cuda()
                batch_size = 0
                tbl = batch['id']
                losses = calculate_cost_online_eval(predictions,embeddings,offsets,displacements,bw, criterion,batch)
                regression_loss += losses['regression_loss'].item()
                embedding_loss += losses['embedding_loss'].item()
                semantic_loss += losses['semantic_loss'].item()
                displacement_loss += losses['displacement_loss'].item()
                classification_loss += losses['classification_loss'].item()
                drift_loss += losses['drift_loss'].item()
                instance_iou += losses['instance_iou'].item()
                predictions = predictions.cpu()
                store.index_add_(0, batch['point_ids'], predictions)
                #visualize based on predictions, use open3D for convinence
                if config['evaluate']:
                    visualize_point_cloud(batch,predictions)
#                            open3d.visualization.draw_geometries([pcd_ori, pcd_gt_label, pcd_predict_label])

                # loop for all val set every snap shot is tooooo slow, pls use val.py to check individually
                # if save_ply:
                #     print("evaluate data: ", i)
                #     iou.visualize_label(batch, predictions, rep, save_dir=config['checkpoints_dir'] +)
            predLabels = store.max(1)[1].numpy()
            topic = 'valid_' + str(rep)
            iou_evaluate(predLabels, valLabels, train_writer, global_iter, class_num = config['class_num'] , topic=topic)
            train_writer.add_scalar(topic + "/epoch_avg_displacement_closs",  displacement_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_regression_closs",  regression_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_embedding_closs", embedding_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_semantic_closs", semantic_loss/ epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_classification_closs", classification_loss / epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_drift_closs", drift_loss/ epoch_len, global_step=global_iter)
            train_writer.add_scalar(topic + "/epoch_avg_instance_precision", instance_iou/ epoch_len, global_step=global_iter)

            print('infer', rep, 'time=', time.time() - time_val_start, 's')
            if config['evaluate']:
                savemat('pred.mat', {'p':store.numpy(),'val_offsets':np.hstack(valOffsets), 'v': valLabels})
            torch.cuda.empty_cache()

        predLabels = store.max(1)[1].numpy()
        iou_evaluate(predLabels, valLabels, train_writer, global_iter,class_num = config['class_num'] ,  topic='valid')
        train_writer.add_scalar("valid/time", time.time() - time_val_start, global_step=global_iter)



def train_net(net, config):
    if config['optim'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'])
    elif config['optim'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=config['lr'])
#        optimizer = optim.Adam([{'params': net.fc_bw.parameters()}, {'params':net.linear_bw.parameters()}], lr=config['lr'])



    """
    if config['loss'] == 'cross_entropy':
        criterion = nn.functional.cross_entropy
    elif config['loss'] == 'focal':
        criterion = FocalLoss()
    elif config['loss'] == 'weighted_cross_entropy':
        if config['dataset'] == 'stanford3d':
            weight = torch.from_numpy(np.hstack([0.1861, 0.1586,0.2663,0.0199,0.0039,0.0210,0.0210,0.0575,0.0332,0.0458,0.0052,0.0495 ,0.0123,0.1164,0.0032]))
        elif config['dataset'] == 'scannet':
            weight = torch.from_numpy(np.hstack([0.3005,0.2700,0.0418,0.0275,0.0810,0.0254,0.0462,0.0418,0.0297,0.0277,0.0061,0.0065,0.0194,0.0150,0.0060,0.0036,0.0029,0.0025,0.0029,0.0434]))
        weight = weight.cuda().float()
        criterion = WeightedCrossEntropyLoss(weight)
    else:
        raise NotImplementedError
    """
    criterion = {}
    criterion['discriminative'] = DiscriminativeLoss(
        DISCRIMINATIVE_DELTA_D,
        DISCRIMINATIVE_DELTA_V
    )
    weight = None
    criterion['nll'] = nn.functional.cross_entropy
    criterion['regression'] = nn.L1Loss()
    for epoch in range(config['checkpoint'], config['max_epoch']):
        net.train()
        stats = {}
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        start = time.time()
        train_loss = 0
        semantic_loss = 0
        regression_loss = 0
        embedding_loss = 0
        displacement_loss = 0
        classification_loss = 0
        instance_iou = 0
        drift_loss = 0
        occupancy_loss = 0
        epoch_len = len(config['train_data_loader'])

        pLabel = []
        tLabel = []
        for i, batch in enumerate(tqdm((config['train_data_loader']))):
            # checked
            # logger.debug("CHECK RANDOM SEED(torch seed): sample id {}".format(batch['id']))
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            batch['x'][1] = batch['x'][1].cuda()
            # print(batch['pth_file'])
            logits, feature, embeddings, offset, displacements, bw, occupancy = net(batch['x'])

            batch['y'] = batch['y'].cuda()
            batch['region_masks'] =  batch['region_masks'].cuda()
            batch['offsets'] =  batch['offsets'].cuda()
            batch['displacements'] =  batch['displacements'].cuda()


            losses = calculate_cost(logits, embeddings, offset, displacements, bw, criterion, batch, occupancy)
            #classification loss


#            loss = losses['semantic_loss'] + losses['regression_loss'] + losses['classification_loss'] + losses['occupancy_loss']

            loss = losses['semantic_loss'] + losses['regression_loss'] + losses['embedding_loss'] + losses['displacement_loss'] + losses['classification_loss'] + losses['occupancy_loss']
            semantic_loss += losses['semantic_loss'].item()
            regression_loss += losses['regression_loss'].item()
            embedding_loss += losses['embedding_loss'].item()
            displacement_loss += losses['displacement_loss'].item()
            classification_loss += losses['classification_loss'].item()
            occupancy_loss += losses['occupancy_loss'].item()
            drift_loss += losses['drift_loss'].item()
            instance_iou += losses['instance_iou'].item()
#            print(losses['drift_loss'].item())

            if i % 50 == 49:
                print('loss: %.2f' % loss.item())

            train_writer.add_scalar("train/loss", loss.item(), global_step=epoch_len * epoch + i)
            predict_label = logits.cpu().max(1)[1].numpy()
            true_label = batch['y'][:,0].detach()
            pLabel.append(torch.from_numpy(predict_label))
            tLabel.append(true_label)
            train_loss += loss.item()

#            memory_used = torch.cuda.memory_allocated(device=None) / torch.tensor(1024*1024*1024).float()
#            print("before backward: ", memory_used)
            loss.backward()
            optimizer.step()
            del loss,losses

#            memory_used = torch.cuda.memory_allocated(device=None)/ torch.tensor(1024*1024*1024).float()
#            print("after backward: ", memory_used)

        torch.cuda.empty_cache()
        pLabel = torch.cat(pLabel,0).cpu().numpy()
        tLabel = torch.cat(tLabel,0).cpu().numpy()
        mIOU = iou_evaluate(pLabel, tLabel, train_writer ,epoch,class_num = config['class_num'] ,  topic = 'train')
        train_writer.add_scalar("train/epoch_avg_loss", train_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_embedding_loss", embedding_loss/ epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_semantic_loss", semantic_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_displacement_loss", displacement_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_regression_loss", regression_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_classification_loss", classification_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_occupancy_loss", occupancy_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_drift_loss", drift_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_instance_precision", instance_iou / epoch_len, global_step= (epoch + 1))

        train_writer.add_scalar("train/time", time.time() - start, global_step=(epoch + 1))
        train_writer.add_scalar("train/lr", config['lr'], global_step=(epoch + 1))
        print(epoch, 'Train loss', train_loss / (i + 1),'/  ',mIOU, 'MegaMulAdd=',
                    scn.forward_pass_multiplyAdd_count / len(dataset.train) /
                    1e6, 'MegaHidden', scn.forward_pass_hidden_states / len(dataset.train) / 1e6, 'time=',
                    time.time() - start, 's')

        # evaluate every config['snapshot'] epoch, and save model at the same time.
        if ((epoch + 1) % config['snapshot'] == 0) or (epoch in [0,4]):
            # net.eval()
#            memory_used = torch.cuda.memory_allocated(device=None)/ torch.tensor(1024*1024*1024).float()
#            print("before empty cache: ", memory_used)
            torch.cuda.empty_cache()
#            memory_used = torch.cuda.memory_allocated(device=None)/ torch.tensor(1024*1024*1024).float()
#            print("after empty cache: ", memory_used)
            evaluate(net=net, config=config, global_iter=(epoch + 1))
            torch.save(net.state_dict(), config['checkpoints_dir'] + 'Epoch{}.pth'.format(epoch + 1))

        if config['gamma'] != 0 and (epoch + 1) % config['step_size'] == 0:
            config['lr'] = config['lr'] * config['gamma']
            if config['optim'] == 'SGD':
                optimizer = optim.SGD(net.parameters(), lr=config['lr'] * config['gamma'], momentum=0.9,
                                      weight_decay=0.0005)
            elif config['optim'] == 'Adam':
                optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.00005)
#                optimizer = optim.Adam([{'params': net.fc_bw.parameters()}, {'params':net.linear_bw.parameters()}], lr=config['lr'], weight_decay=0.00005)
        # if scn.is_power2(epoch) or (epoch % eval_epoch == 0 if eval_epoch else False) or epoch == training_epochs:
        #         evaluate(unet,valOffsets,val_data_loader,valLabels,save_ply=(eval_save_ply and scn.is_power2(epoch)),
        #                  prefix="epoch_{epoch}_".format(epoch=epoch))


def train_uncertain(net, config):
    if config['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=config['lr'])
    elif config['optim'] == 'Adam':
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=config['lr'])
#        optimizer = optim.Adam([{'params': net.fc_bw.parameters()}, {'params':net.linear_bw.parameters()}], lr=config['lr'])



    """
    if config['loss'] == 'cross_entropy':
        criterion = nn.functional.cross_entropy
    elif config['loss'] == 'focal':
        criterion = FocalLoss()
    elif config['loss'] == 'weighted_cross_entropy':
        if config['dataset'] == 'stanford3d':
            weight = torch.from_numpy(np.hstack([0.1861, 0.1586,0.2663,0.0199,0.0039,0.0210,0.0210,0.0575,0.0332,0.0458,0.0052,0.0495 ,0.0123,0.1164,0.0032]))
        elif config['dataset'] == 'scannet':
            weight = torch.from_numpy(np.hstack([0.3005,0.2700,0.0418,0.0275,0.0810,0.0254,0.0462,0.0418,0.0297,0.0277,0.0061,0.0065,0.0194,0.0150,0.0060,0.0036,0.0029,0.0025,0.0029,0.0434]))
        weight = weight.cuda().float()
        criterion = WeightedCrossEntropyLoss(weight)
    else:
        raise NotImplementedError
    """
    criterion = {}
    criterion['discriminative'] = DiscriminativeLoss(
        DISCRIMINATIVE_DELTA_D,
        DISCRIMINATIVE_DELTA_V
    )
    weight = None
    criterion['nll'] = nn.functional.cross_entropy
    criterion['regression'] = nn.L1Loss()
    if config['bceloss'] == 'weighted_bce':
        criterion['binnary_classification'] = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['uncertain_weight']]).cuda())
    elif config['bceloss'] == 'focal_loss':
        criterion['binnary_classification'] = bcelosses.BCEFocalLoss(alpha=config['uncertain_weight']/(config['uncertain_weight'] + 1))
    elif config['bceloss'] == 'lovasz':
        criterion['binnary_classification'] = lovasz_hinge
    for epoch in range(config['checkpoint'], config['max_epoch']):
        net.train()
        stats = {}
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        start = time.time()
        train_loss = 0
        semantic_loss = 0
        regression_loss = 0
        embedding_loss = 0
        displacement_loss = 0
        classification_loss = 0
        instance_iou = 0
        drift_loss = 0
        uncertain_loss = 0
        consistent_loss = 0
        loss_var = 0
        loss_dis = 0
        loss_reg = 0
        epoch_len = len(config['train_data_loader'])
        cum_loss = 0
        pLabel = []
        tLabel = []
        uncertain_fp = 0
        uncertain_fn = 0
        uncertain_tp = 0
        uncertain_tn = 0
        uncertain_num = 0
        tot_num = 0
        consistency_percent = 0
        emb_instance_precision = 0
        print('TASK NAME =', config['taskname'])
        for i, batch in enumerate(tqdm((config['train_data_loader']))):
            # checked
            # logger.debug("CHECK RANDOM SEED(torch seed): sample id {}".format(batch['id']))
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            batch['x'][1] = batch['x'][1].cuda()
            # print(batch['pth_file'])
            logits, feature, embeddings, offset, displacements, bw, uncertain = net(batch['x'])

            batch['y'] = batch['y'].cuda()
            batch['region_masks'] =  batch['region_masks'].cuda()
            batch['offsets'] =  batch['offsets'].cuda()
            batch['displacements'] =  batch['displacements'].cuda()
            try:
                torch.cuda.empty_cache()
                losses = calculate_cost_online(logits, embeddings, offset, displacements, bw, criterion, batch, uncertain, epoch, config)
            #classification loss

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise exception


#            loss = losses['semantic_loss'] + losses['regression_loss'] + losses['classification_loss'] + losses['occupancy_loss']

            if losses['consistent_loss'] > 10:
                print('Consistency loss out of bound:', losses['consistent_loss'])
                #losses['consistent_loss'] = torch.clamp(losses['consistent_loss'], max=10.0)

            if losses['uncertain_loss'] > 3:
                print('Uncertain Loss out of bound:', losses['uncertain_loss'])
                losses['uncertain_loss'] *= 0.2


            loss = losses['semantic_loss'] + losses['regression_loss'] + losses['embedding_loss'] + losses['displacement_loss'] + losses['classification_loss'] + losses['uncertain_loss'] + losses['consistent_loss']
            semantic_loss += losses['semantic_loss'].item()
            regression_loss += losses['regression_loss'].item()
            embedding_loss += losses['embedding_loss'].item()
            displacement_loss += losses['displacement_loss'].item()
            classification_loss += losses['classification_loss'].item()
            
            loss_var += losses['emb_var'].item()
            loss_dis += losses['emb_dis'].item()
            loss_reg += losses['emb_reg'].item()

            uncertain_loss += losses['uncertain_loss'].item()
            

            drift_loss += losses['drift_loss'].item()
            instance_iou += losses['instance_iou'].item()

            emb_instance_precision += losses['emb_instance_precision']
            
            consistent_loss += losses['consistent_loss'].item()
#            print(losses['drift_loss'].item())
            cum_loss += loss.item()
            uncertain_tp += losses['uncertain_tp']
            uncertain_tn += losses['uncertain_tn']
            uncertain_fp += losses['uncertain_fp']
            uncertain_fn += losses['uncertain_fn']
            uncertain_num += losses['uncertain_num']
            tot_num += losses['tot_num']
            
            consistency_percent += losses['consistency_percent']

            if i % 50 == 49:
                print('loss: %.2f' % (cum_loss / (i + 1)))

            # train_writer.add_scalar("train/loss", loss.item(), global_step=epoch_len * epoch + i)
            predict_label = logits.cpu().max(1)[1].numpy()
            true_label = batch['y'][:,0].detach().cpu()
            pLabel.append(torch.from_numpy(predict_label))
            tLabel.append(true_label)
            train_loss += loss.item()

#            memory_used = torch.cuda.memory_allocated(device=None) / torch.tensor(1024*1024*1024).float()
#            print("before backward: ", memory_used)
            loss.backward()
            optimizer.step()
            del loss,losses

#            memory_used = torch.cuda.memory_allocated(device=None)/ torch.tensor(1024*1024*1024).float()
#            print("after backward: ", memory_used)

        #torch.cuda.empty_cache()
        pLabel = torch.cat(pLabel,0).numpy()
        tLabel = torch.cat(tLabel,0).numpy()
        
        
        mIOU = iou_evaluate(pLabel, tLabel, train_writer ,epoch,class_num = config['class_num'] ,  topic = 'train')
        train_writer.add_scalar("train/epoch_avg_loss", train_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_embedding_loss", embedding_loss/ epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_semantic_loss", semantic_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_displacement_loss", displacement_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_regression_loss", regression_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_classification_loss", classification_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_uncertain_loss", uncertain_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_consistent_loss", consistent_loss / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_instance_precision", instance_iou / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_consistency_percent", consistency_percent / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_emb_instance_precision", emb_instance_precision / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_embedding_var", loss_var / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_embedding_dis", loss_dis / epoch_len, global_step= (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_embedding_reg", loss_reg / epoch_len, global_step= (epoch + 1))
        
        if epoch >= config['uncertain_st_epoch']:
            uncertain_iou = uncertain_tp / (uncertain_tp + uncertain_fn + uncertain_fp)
            try:
                train_writer.add_scalar("train/epoch_avg_uncertain_accuracy", (uncertain_tp + uncertain_tn) / (uncertain_tp + uncertain_tn + uncertain_fp + uncertain_fn), global_step= (epoch + 1))
                train_writer.add_scalar("train/epoch_avg_uncertain_precision", uncertain_tp / (uncertain_tp + uncertain_fp), global_step= (epoch + 1))
                train_writer.add_scalar("train/epoch_avg_uncertain_recall", uncertain_tp / (uncertain_tp + uncertain_fn), global_step= (epoch + 1))
                train_writer.add_scalar("train/epoch_avg_uncertain_iou", uncertain_iou, global_step= (epoch + 1))
                train_writer.add_scalar("train/epoch_avg_uncertain_percent", uncertain_num / tot_num, global_step= (epoch + 1))
            except ZeroDivisionError:
                print('division by zero!')
                print('uncertain_fp = %d\nuncertain_fn = %d\nuncertain_tp = %d\nuncertain_tn = %d\nuncertain_num = %d\ntot_num = %d' % (uncertain_fp, uncertain_fn, uncertain_tp, uncertain_tn, uncertain_num, tot_num))

        train_writer.add_scalar("train/time", time.time() - start, global_step=(epoch + 1))
        train_writer.add_scalar("train/lr", config['lr'], global_step=(epoch + 1))
        print(epoch, 'Train loss', train_loss / (i + 1),'/  ',mIOU, 'MegaMulAdd=',
                    scn.forward_pass_multiplyAdd_count / len(dataset.train) /
                    1e6, 'MegaHidden', scn.forward_pass_hidden_states / len(dataset.train) / 1e6, 'time=',
                    time.time() - start, 's')

        # evaluate every config['snapshot'] epoch, and save model at the same time.
        if ((epoch + 1) % config['snapshot'] == 0) or (epoch in [0,4]):
            # net.eval()
#            memory_used = torch.cuda.memory_allocated(device=None)/ torch.tensor(1024*1024*1024).float()
#            print("before empty cache: ", memory_used)
            torch.cuda.empty_cache()
#            memory_used = torch.cuda.memory_allocated(device=None)/ torch.tensor(1024*1024*1024).float()
#            print("after empty cache: ", memory_used)
            evaluate_online(net=net, config=config, global_iter=(epoch + 1))
            torch.save(net.state_dict(), config['checkpoints_dir'] + 'Epoch{}.pth'.format(epoch + 1))

        if config['gamma'] != 0 and (epoch + 1) % config['step_size'] == 0:
            config['lr'] = config['lr'] * config['gamma']
            if config['optim'] == 'SGD':
                optimizer = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=config['lr'] * config['gamma'], momentum=0.9,
                                      weight_decay=0.0005)
            elif config['optim'] == 'Adam':
                optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=config['lr'], weight_decay=0.00005)
#                optimizer = optim.Adam([{'params': net.fc_bw.parameters()}, {'params':net.linear_bw.parameters()}], lr=config['lr'], weight_decay=0.00005)
        # if scn.is_power2(epoch) or (epoch % eval_epoch == 0 if eval_epoch else False) or epoch == training_epochs:
        #         evaluate(unet,valOffsets,val_data_loader,valLabels,save_ply=(eval_save_ply and scn.is_power2(epoch)),
        #                  prefix="epoch_{epoch}_".format(epoch=epoch))




def preprocess():
    args = get_args()
    config = ArgsToConfig(args)

    torch.cuda.set_device(args.gpu)

    torch.manual_seed(100)  # cpu
    torch.cuda.manual_seed(100)  # gpu
    np.random.seed(100)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    # setup config
    # choose kernel size
    # if config['kernel_size'] == 3:
    #     Model = ThreeVoxelKernel
    # else:
    #     raise NotImplementedError

    if config['model_type'] == 'occ':
        Model = LearningBWDenseUNet
    elif config['model_type'] == 'uncertain':
        Model = UncertainDenseUNet




    train_writer = SummaryWriter(comment=args.taskname)

    if os.path.exists(config['checkpoints_dir']) is False:
        os.makedirs(config['checkpoints_dir'])


    # choose dataset
    if args.dataset == 'scannet':
        config['class_num'] = 20
        iou_evaluate = evaluate_scannet
        dataset_dir = 'scannet_data/instance'
    elif args.dataset == 'stanford3d':
        config['class_num'] = 14
        iou_evaluate = evaluate_stanford3D
        dataset_dir = 'stanford_data/instance'
    else:
        raise NotImplementedError

    # combine multiple data sources
    use_train_data = args.use_train_data
    use_val_data = args.use_val_data


    if True:
        if args.all_to_train == True:
            train_dataset_mid_dir = 'full_train'
            test_dataset_mid_dir = 'full_val'
        else:
            train_dataset_mid_dir = 'train'
            test_dataset_mid_dir = 'val'

        if args.evaluate:
            if args.all_to_train == True:
                train_dataset_mid_dir = 'full_train'
                test_dataset_mid_dir = 'test'
            else:
                train_dataset_mid_dir = 'train'
                test_dataset_mid_dir = 'val'

        pth_reg_exp = '*.pth'

        train_pth_path=[]
        if args.test:
            train_pth_path.append('datasets/simple_data/*.pth')
            val_pth_path = 'datasets/simple_data/*.pth'
        else:
            if 'o' in use_train_data:
                train_pth_path.append('datasets/{}/{}/{}'.format(dataset_dir, train_dataset_mid_dir, pth_reg_exp))
    #        if '0' in use_train_data:
    #            train_pth_path.append('datasets/{}/{}/{}'.format(dataset_dir, "partial_0" ,train_dataset_mid_dir, pth_reg_exp))
    #        if '1' in use_train_data:
    #            train_pth_path.append('datasets/{}/{}/{}'.format(dataset_dir, "partial_1" ,train_dataset_mid_dir, pth_reg_exp))
    #        if '2' in use_train_data:
    #            train_pth_path.append('datasets/{}/{}/{}'.format(dataset_dir, "partial_2" ,train_dataset_mid_dir, pth_reg_exp))

            if 'o' in use_val_data:
                val_pth_path='datasets/{}/{}/{}'.format(dataset_dir, test_dataset_mid_dir, pth_reg_exp)
    #        elif '0' in use_val_data:
    #            val_pth_path='datasets/{}/{}/{}/{}'.format(dataset_dir, "partial_0", test_dataset_mid_dir, pth_reg_exp)
    #        else:
    #            raise


        if args.dataset == 'stanford3d':
            pth_reg_exp = '*.pth'
            test_scene = 5
            train_pth_path = []
            for candidate_sence in range(1,7):
                if(candidate_sence == test_scene):
                    val_pth_path    =   'datasets/{}/{}/{}'.format(dataset_dir, 'full' , 'Area_'+str(test_scene)+'_*.pth')
                else:
                    train_pth_path.append('datasets/{}/{}/{}'.format(dataset_dir, 'full' , 'Area_'+str(candidate_sence)+'_*.pth'))

        if config['simple_train'] == True:
            train_pth_path='datasets/simple_data/{}'.format(pth_reg_exp)
            val_pth_path='datasets/simple_data/{}'.format(pth_reg_exp)
        print(train_pth_path, val_pth_path)
        if config['model_type'] == 'occ':
            dataset = ScanNet(train_pth_path=train_pth_path,
                            val_pth_path=val_pth_path,
                            config = config,
                            )
        elif config['model_type'] == 'uncertain':
            train_pth_path = 'datasets/scannetTrainSeq/*/*_instance.pth'
            dataset = ScanNetOnline(train_pth_path=train_pth_path,
                            val_pth_path=val_pth_path,
                            config = config,
                            train_seq_path='./datasets/scannetTrainSeq'
                            )

            


    # log the config to tensorboard
    tmp_config_str = ''
    for k, v in config.items():
        if isinstance(v, str) or isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) or isinstance(v, int):
            train_writer.add_text('config', '{:<8}   :    {:>8}\n'.format(k, v), global_step=0)
        else:
            train_writer.add_text('config', '{:<8}   :    {:>8}\n'.format(k, json.dumps(v)), global_step=0)

    config['valOffsets'], \
    config['train_data_loader'], \
    config['val_data_loader'], \
    config['valLabels'] = dataset.load_data()

    net = Model(config)
    print('#classifer parameters', sum([x.nelement() for x in net.parameters()]))

    if config['pretrain'] != 'none':
        net.load_my_pretrain(config['pretrain'])
        print('load pretrain from', config['pretrain'])

    if config['model_type'] == 'uncertain':
       # assert(config['pretrain'] != 'none')
        if config['freeze_type'] == 'unet':
            net.freeze_unet()
        elif config['freeze_type'] == 'unetex4':
            net.freeze_except_unet_4()
        elif config['freeze_type'] == 'backbone':
            net.freeze_backbone()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        config['checkpoint'] = 0
        print('Model loaded from {}'.format(args.load))


    if config['restore']:
        pre_epoch = -1
        for pth in os.listdir(config['checkpoints_dir']):
            pre_epoch = max(pre_epoch, int(pth[pth.find('Epoch')+5:pth.find('.pth')]))
        if pre_epoch != -1:
            net.load_state_dict(torch.load(os.path.join(config['checkpoints_dir'], 'Epoch' + str(pre_epoch) + '.pth')))
            print('Model restored from epoch:', pre_epoch)
            config['checkpoint'] = pre_epoch



    """
    i = 0
    print("load model from: ", args.load)
    print("Saving weights to npy files...")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        np.save('weight/unet_' + str(i) + '.npy', net.state_dict()[param_tensor].cpu().detach().numpy())
        i = i + 1
    raise ValueError("Model Loaded")
    """
    # pls use GPU to train
    assert torch.cuda.is_available()
    net.cuda()

    for key in config.keys():
        print(key,[config[key]])
    return args, config, net, dataset, iou_evaluate, train_writer


if __name__ == '__main__':


    args, config, net, dataset, iou_evaluate,train_writer  = preprocess()
    try:

        if(args.evaluate):
            evaluate_instance(net=net,  config=config,global_iter=0)
        elif config['model_type'] == 'occ':
            train_net(net=net, config=config)
        elif config['model_type'] == 'uncertain':
            train_uncertain(net, config)


    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
