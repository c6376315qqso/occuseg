import torch
import torch.nn as nn
import pdb
import math
from torch_scatter import scatter_max,scatter_mean,scatter_std,scatter_sub,scatter_min,scatter_add,scatter_div

DISCRIMINATIVE_DELTA_V = 0.2
DISCRIMINATIVE_DELTA_D = 1.5

def DriftLoss(embedded, masks, pred_semantics, regressed_pose, offset, pose):
    batch_size = embedded.size(0)
    loss = torch.zeros(1,dtype=torch.float32).cuda()
    mask_count = 0
    criterion = nn.L1Loss()
    for i in range(batch_size):
        mask_size = masks.shape[2]
        for mid in range(mask_size):
            instance_indices = (masks[i,:,mid]==1)
            cls = pred_semantics[instance_indices][0]
            if(cls > 1):
                mu = embedded[i,instance_indices,:].mean(dim =0)
                mean_pose = pose[i,instance_indices,:].mean(dim = 0)
                semantic_embedding = embedded[i,instance_indices,:]
                spatial_embedding = regressed_pose[i,instance_indices,:]
                weight = offset[instance_indices,:]
                valid_index = (weight > 0.01).view(-1).detach()
                if(valid_index.sum(0) > 0):
                    weight = weight[valid_index,:]
                    semantic_embedding = semantic_embedding[valid_index ,:]
                    spatial_embedding = spatial_embedding[valid_index ,:]
                    semantic_embedding = semantic_embedding * weight.expand_as(semantic_embedding)
                    spatial_embedding = spatial_embedding * weight.expand_as(spatial_embedding)
                    semantic_target = mu.expand_as(semantic_embedding) * weight.expand_as(semantic_embedding)
                    spatial_target = mean_pose.expand_as(spatial_embedding) * weight.expand_as(spatial_embedding)
                    loss += criterion(semantic_embedding, semantic_target) + criterion(spatial_embedding , spatial_target)
                mask_count += 1
    if(mask_count > 0):
        loss = loss / mask_count
    return loss


# make the inputs distinguishable compared with previous methods
def ClassificationLoss(embedded, bw, regressed_pose,pose,instance_mask,pred_semantics):
    batch_size = embedded.size(0)
    loss = torch.zeros(1,dtype=torch.float32).cuda()
    mask_count = 0
    tp = 0
    total = 1
    fp = 0
    miou = 0

    # occlusion matters in 2D!!
    for i in range(batch_size):
        volume = torch.cat((embedded[i,:,:], pose[i,:,:],bw[i,:,:],regressed_pose[i,:,:]),dim = 1)
        instance_mean_volume = scatter_mean(volume,instance_mask[i,:],dim = 0)
        bw_std = scatter_std(bw[i,:,:],instance_mask[i,:],dim = 0)
#        bw_std = torch.clamp(bw_std ** 2 - 0.01 , min = 0.0)
#        pose_std = scatter_std(pose[i,:,:],instance_mask[i,:],dim = 0).norm(dim=1).view(-1)
        mask_size = instance_mask[i,:].max() + 1
        for mid in range(mask_size):
            instance_indices = (instance_mask[i,:]==mid)
            if instance_indices.sum(0) == 0:
                continue
            cls = pred_semantics[instance_indices][0]
            if(cls > -1):
                instance_indices = (instance_mask[i,:]==mid)
                if(instance_indices.sum(0) < 30):
                    continue
                start = 0
                end = embedded.shape[2]
                mu = instance_mean_volume[mid,start:end]  # mean feature term R^256
                start += embedded.shape[2]                  
                end += pose.shape[2]
                mean_pose = instance_mean_volume[mid,start:end] # pose, predicted instance center R^3
                start += pose.shape[2]
                end += 1
                sigma1 = instance_mean_volume[mid,start:end]  # mean covariance 0
                start += 1
                end += 1
                sigma2 = instance_mean_volume[mid,start:end]  # mean covariance 1

                ######## choose 4 * max instance radius to test precision in order to reduce computation burden.
                spatial_distance = (pose - mean_pose.expand_as(pose)).norm(dim=2).view(-1)
                dist_threshold = torch.max(spatial_distance[instance_indices]) * 4
                samples = spatial_distance < dist_threshold
                samples_gt = instance_indices[samples]
                semantic_embedding = embedded[i,samples,:]
                spatial_embedding = regressed_pose[i,samples,:]
                dist1 = semantic_embedding - mu.expand_as(semantic_embedding)
                dist2 = spatial_embedding - mean_pose.expand_as(spatial_embedding)
                d1 = dist1.norm(dim = 1) * sigma1
                d2 = dist2.norm(dim = 1) * sigma2
                prob = torch.exp(-d1*d1-d2*d2)
#                prob = torch.exp(-d1*d1)
#                prob = torch.exp(-(dist1.norm(dim = 1) / sigma1) **  2 - (dist2.norm(dim=1) / (sigma2+1)) ** 2)
#                print(dist1.norm(dim = 1),dist2.norm(dim=1) )
                # control deviations of predicted instance segmentation
#                bw_devation = (bw[i,instance_indices,:] - torch.cat([sigma1, sigma2]).view(1,2)).norm(2,dim = 1)
#                bw_devation = torch.mean(torch.clamp(bw_devation - 0.01, min=0.0) ** 2)

#                weight = torch.ones([samples_gt.shape[0]]).cuda()
#                weight[samples_gt] = samples_gt.shape[0] / samples_gt.sum(0)
#                criterion = nn.BCELoss(weight)

                criterion = nn.BCELoss()
                loss += criterion(prob, samples_gt.float()) #+ (bw_std[mid,0] + bw_std[mid,1]) * 0.1
#                print(criterion(prob, samples_gt.float()), bw_devation * 0.01)
                with torch.no_grad():
                    u = prob > 0.5
                    v = samples_gt
                    tp = (u * v).sum(0).item()
                    fp = (u * (~v)).sum(0).item()
                    total = (v.sum(0)).item()
                    miou += tp / (total + fp)
#                    loss += lovasz_softmax(prob.view(-1,1), instance_indices.float(),classes=[1])
#                    loss += criterion(prob, instance_indices.float())
                mask_count += 1
    if(mask_count > 0):
        loss = loss / mask_count * 10
        miou = miou / mask_count
    return loss, miou

#only support batchSize = 1 as different image has different instances
class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_d, delta_v,
                 alpha=1.0, beta=1.0, gamma=0.001,
                 reduction='mean'):
        # TODO: Respect the reduction rule
        super(DiscriminativeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Set delta_d > 2 * delta_v
        self.delta_d = delta_d
        self.delta_v = delta_v

    def forward(self, embedded, instance_mask):
        centroids = self._new_centroids(embedded, instance_mask)
        L_v = self._new_variance(embedded, instance_mask, centroids)
        # instance num
        size = torch.max(instance_mask,dim = 1)[0] + 1
        L_d = self._distance(centroids, size)
        L_r = self._regularization(centroids, size)
        loss = self.alpha * L_v + self.beta * L_d + self.gamma * L_r
        return loss, L_v, L_d, L_r



    def _new_centroids(self, embedded, instance_mask):

        batch_size = embedded.size(0)
        centroids = []
        for i in range(batch_size):
            mu = scatter_mean(embedded[i,:,:],instance_mask[i,:],dim=0)
            centroids.append(mu)
        centroids = torch.stack(centroids)
        return centroids

    def _centroids(self, embedded, masks, size):
        batch_size = embedded.size(0)
        embedding_size = embedded.size(2)
        K = masks.size(2)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)
        masks = masks.unsqueeze(3)
#        pdb.set_trace()
        x = x * masks
        centroids = []
        for i in range(batch_size):
            n = size[i]
            mu = x[i,:,:n].sum(0) / masks[i,:,:n].sum(0)
            if K > n:
                m = int(K - n)
                filled = torch.zeros(m, embedding_size)
                filled = filled.to(embedded.device)
                mu = torch.cat([mu, filled], dim=0)
            centroids.append(mu)
        centroids = torch.stack(centroids)
        return centroids

    def _new_variance(self, embedded, instance_mask, centroids):

        batch_size = embedded.size(0)
        loss = 0.0
        for i in range(batch_size):
            devation =  (embedded[i,:,:] - torch.index_select(centroids[i,:,:],0,instance_mask[i,:].view(-1))).norm(2,dim = 1)
            devation = torch.clamp(devation - self.delta_v, min=0.0) ** 2
            # shall we include an weight inversely propotional to instance size, to balance training?   
            loss += devation.mean()
        return loss


    def _variance(self, embedded, masks, centroids, size):
        batch_size = embedded.size(0)
        num_points = embedded.size(1)
        embedding_size = embedded.size(2)
        K = masks.size(2)
        # Convert input into the same size
        mu = centroids.unsqueeze(1).expand(-1, num_points, -1, -1)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)
        # Calculate intra pull force
        var = torch.norm(x - mu, 2, dim=3)
        var = torch.clamp(var - self.delta_v, min=0.0) ** 2
        var = var * masks
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            loss += torch.sum(var[i,:,:n]) / torch.sum(masks[i,:,:n])
        return loss

    def _distance(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            if n <= 1: continue
            mu = centroids[i, :n, :]
            mu_a = mu.unsqueeze(1).expand(-1, n, -1)
            mu_b = mu_a.permute(1, 0, 2)
            diff = mu_a - mu_b
            norm = torch.norm(diff, 2, dim=2)
            margin = 2 * self.delta_d * (1.0 - torch.eye(n))
            margin = margin.to(centroids.device)
            distance = torch.sum(torch.clamp(margin - norm, min=0.0) ** 2) # hinge loss
            distance /= float(n * (n - 1))
            loss += distance
        loss /= batch_size
        return loss

    def _regularization(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            mu = centroids[i, :n, :]
            norm = torch.norm(mu, 2, dim=1)
            loss += torch.mean(norm)
        loss /= batch_size
        return loss




def ConsistencyLoss_p2i(embeddings, indexs, instance_masks, max_instance_id, instance_sizes, instance_cls, num_per_scene):
    consistent_num = 0
    complete_id = num_per_scene - 1
    comp_mean_embeddings = scatter_mean(embeddings[indexs[complete_id]], instance_masks[indexs[complete_id]], dim=0)
    loss = torch.zeros(1).cuda()
    points_num = 0
    complete_num = torch.sum(indexs[complete_id])
    for partial_id in range(num_per_scene - 1):
        index = indexs[partial_id]
        if (torch.sum(index) == complete_num):
            continue
        embedding = embeddings[index]
        instance_mask = instance_masks[index]
        for instance_id in range(max_instance_id):
            mask = instance_mask == instance_id
            if instance_sizes[partial_id, instance_id] > 30 and instance_cls[instance_id] > 1:
                norm = torch.norm(embedding[mask] - comp_mean_embeddings[instance_id], 2, dim=1)
                var = torch.clamp(norm - DISCRIMINATIVE_DELTA_V, min=0.0) ** 2
                # shall we include an weight inversely propotional to instance size , to balance training?   
                loss += var.sum()
                consistent_num += torch.sum(norm < DISCRIMINATIVE_DELTA_V).item()
                points_num += norm.shape[0]
    if points_num > 0:
        loss /= points_num
        consistent_num /= points_num
    return loss, consistent_num
    





def ConsistencyLoss_i2i(embeddings, indexs, instance_masks, max_instance_id, instance_sizes, instance_cls, num_per_scene):
    consistent_num = 0
    complete_id = num_per_scene - 1
    comp_mean_embeddings = scatter_mean(embeddings[indexs[complete_id]], instance_masks[indexs[complete_id]], dim=0)
    loss = torch.zeros(1).cuda()
    points_num = 0
    ins_num = 0
    complete_num = torch.sum(indexs[complete_id])

    for partial_id in range(num_per_scene - 1):
        index = indexs[partial_id]
        if (torch.sum(index) == complete_num):
            continue
        embedding = embeddings[index]
        instance_mask = instance_masks[index]
        for instance_id in range(max_instance_id):
            mask = instance_mask == instance_id
            if instance_sizes[partial_id, instance_id] > 30 and instance_cls[instance_id] > 1:
                norm = torch.norm(torch.mean(embedding[mask], dim=0) - comp_mean_embeddings[instance_id], 2, dim=0)
                var = torch.clamp(norm - DISCRIMINATIVE_DELTA_V, min=0.0) ** 2
                loss += var
                points_norm = torch.norm(embedding[mask] - comp_mean_embeddings[instance_id], 2, dim=1)
                consistent_num += torch.sum(points_norm < DISCRIMINATIVE_DELTA_V).item()
                points_num += points_norm.shape[0]
                ins_num += 1
    if ins_num > 0:
        loss /= ins_num
    if points_num > 0:
        consistent_num /= points_num
    return loss, consistent_num


def ConsistencyLoss_p2p(embeddings, indexs, instance_masks, max_instance_id, instance_sizes, instance_cls, num_per_scene, scene_masks):
    consistent_num = 0
    complete_id = num_per_scene - 1
    complete_embedding = embeddings[indexs[complete_id]]
    comp_mean_embeddings = scatter_mean(embeddings[indexs[complete_id]], instance_masks[indexs[complete_id]], dim=0)
    loss = torch.zeros(1).cuda()
    points_num = 0
    complete_num = torch.sum(indexs[complete_id])
    for partial_id in range(num_per_scene - 1):
        index = indexs[partial_id]
        if (torch.sum(index) == complete_num):
            continue
        scene_mask = scene_masks[partial_id]
        embedding = embeddings[index]
        instance_mask = instance_masks[index]
        for instance_id in range(max_instance_id):
            mask = instance_mask == instance_id
            if instance_sizes[partial_id, instance_id] > 30 and instance_cls[instance_id] > 1:
                norm = torch.norm(embedding[mask] - complete_embedding[scene_mask][mask], 2, dim=1)
                var = torch.clamp(norm - DISCRIMINATIVE_DELTA_V, min=0.0) ** 2
                loss += var.sum()
                points_norm = torch.norm(embedding[mask] - comp_mean_embeddings[instance_id], 2, dim=1).detach()
                consistent_num += torch.sum(points_norm < DISCRIMINATIVE_DELTA_V).item()
                points_num += norm.shape[0]
    if points_num > 0:
        loss /= points_num
        consistent_num /= points_num
    return loss, consistent_num



def Consistency_Evaler(embeddings, indexs, instance_masks, max_instance_id, instance_sizes, instance_cls, num_per_scene):
    with torch.no_grad():
        consistent_num = 0.0
        complete_id = num_per_scene - 1
        comp_mean_embeddings = scatter_mean(embeddings[indexs[complete_id]], instance_masks[indexs[complete_id]], dim=0)
        points_num = 0
        complete_num = torch.sum(indexs[complete_id])
        for partial_id in range(num_per_scene - 1):
            index = indexs[partial_id]
            if (torch.sum(index) == complete_num):
                continue
            embedding = embeddings[index]
            instance_mask = instance_masks[index]
            for instance_id in range(max_instance_id):
                mask = instance_mask == instance_id
                if instance_sizes[partial_id, instance_id] > 30 and instance_cls[instance_id] > 1:
                    norm = torch.norm(embedding[mask] - comp_mean_embeddings[instance_id], 2, dim=1)
                    consistent_num += torch.sum(norm < DISCRIMINATIVE_DELTA_V).item()
                    points_num += norm.shape[0]
        if points_num > 0:
            consistent_num /= points_num
    return consistent_num


def Embedding_Evaler(embeddings, indexs, instance_masks, max_instance_id, instance_sizes, instance_cls, num_per_scene, poses):
    with torch.no_grad():
        instance_cnt = 0
        miou = 0
        for partial_id in range(num_per_scene):
            index = indexs[partial_id]
            instance_mask = instance_masks[index]
            pose = poses[index]
            embedding = embeddings[index]
            mean_pose = scatter_mean(pose, instance_mask, dim=0)
            mean_embedding = scatter_mean(embedding, instance_mask, dim=0)

            for i in range(max_instance_id):
                if instance_sizes[partial_id, i] <= 30 or instance_cls[i] <= 1:
                    continue
                instance_indices = instance_mask == i
                ######## choose 4 * max instance radius to test precision in order to reduce computation burden.
                spatial_distance = (pose - mean_pose[i]).norm(dim=1)
                dist_threshold = torch.max(spatial_distance[instance_indices]) * 4
                samples = spatial_distance < dist_threshold
                samples_gt = instance_indices[samples]
                samples_embedding = embedding[samples,:]
                
                samples_norm = torch.norm(samples_embedding - mean_embedding[i], 2, dim=1)
                u = samples_norm < DISCRIMINATIVE_DELTA_V 
                v = samples_gt
                tp = (u * v).sum(0).item()
                fp = (u * (~v)).sum(0).item()
                total = (v.sum(0)).item()
                miou += tp / (total + fp)
                instance_cnt += 1
    if instance_cnt > 0:
        miou /= instance_cnt
    return miou



def Consistency_Evaler_i2i(embeddings, indexs, instance_masks, max_instance_id, instance_sizes, instance_cls, num_per_scene):
    with torch.no_grad():
        consistent_num = 0.0
        complete_id = num_per_scene - 1
        comp_mean_embeddings = scatter_mean(embeddings[indexs[complete_id]], instance_masks[indexs[complete_id]], dim=0)
        instance_num = 0
        complete_num = torch.sum(indexs[complete_id])
        for partial_id in range(num_per_scene - 1):
            index = indexs[partial_id]
            if (torch.sum(index) == complete_num):
                continue
            embedding = embeddings[index]
            instance_mask = instance_masks[index]
            for instance_id in range(max_instance_id):
                mask = instance_mask == instance_id
                if instance_sizes[partial_id, instance_id] > 30 and instance_cls[instance_id] > 1:
                    norm = torch.norm(torch.mean(embedding[mask], dim=0) - comp_mean_embeddings[instance_id], 2, dim=0)
                    consistent_num += torch.sum(norm < DISCRIMINATIVE_DELTA_V).item()
                    instance_num += 1
        if instance_num > 0:
            consistent_num /= instance_num
    return consistent_num