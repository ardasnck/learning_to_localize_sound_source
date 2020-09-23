import torch
import torch.nn as nn
import torch.nn.functional as F



class UnsupervisedLoss(nn.Module):
    def __init__(self):
        super(UnsupervisedLoss, self).__init__()

    def forward(self, f_v, f_pos, f_neg, weights, opt):
        d_pos = F.pairwise_distance(f_v, f_pos, keepdim = False)
        d_neg = F.pairwise_distance(f_v, f_neg, keepdim = False)
        dist_vector = torch.stack((d_pos,d_neg),1)
        dist_softmax = F.softmax(dist_vector, dim=-1)
        dist_target = torch.stack((torch.zeros((f_v.size(0))),torch.ones((f_v.size(0)))),1).to(opt.device)
        loss = (dist_softmax - dist_target)**2
        return loss.mean()


class SupervisedLoss(nn.Module):
    def __init__(self):
        super(SupervisedLoss, self).__init__()

    def forward(self,pred_att, gt_att, weights):
        # pred_att size (N,400)
        # gt_att size (N,400)
        # weights size (N,2)
        cross_entropy_loss = -(torch.sum(torch.mul(gt_att,torch.log(pred_att)),1))
        cross_entropy_loss = torch.mul(cross_entropy_loss,weights[:,0])
        return cross_entropy_loss.mean()