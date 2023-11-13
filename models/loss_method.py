import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# from .loss import BCELoss, WBCELoss

def loss_DKD(logit, label, n_old_classes, n_new_classes,
              BCELoss_func, ACLoss_func, 
              KDLoss_func = None, logit_old = None, features = None, features_old = None ):
    if KDLoss_func is None:
        assert logit_old is None and features is None and features_old is None, \
            "KDLoss_func is None but logit_old or features or features_old is not None"
    if logit_old is None and KDLoss_func is None:
        # step 0 loss
        loss_mbce = BCELoss_func(
                    logit[:, -n_new_classes:],  # [N, |Ct|, H, W]
                    label,                # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]
        loss_kd = 0
        loss_ac = ACLoss_func(logit[:, :1]).mean(dim=[0, 2, 3]) # [1]
        loss_dkd_pos = 0
        loss_dkd_neg = 0
    elif logit_old is not None and KDLoss_func is not None:
        # step 1 ~ loss
        # [|Ct|]
        loss_mbce = BCELoss_func(
                    logit[:, -n_new_classes:],  # [N, |Ct|, H, W]
                    label,                # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]
        # [|C0:t-1|]
        loss_kd = KDLoss_func(
            logit[:, 1:n_old_classes + 1],  # [N, |C0:t|, H, W]
            logit_old[:, 1:].sigmoid()       # [N, |C0:t|, H, W]
        ).mean(dim=[0, 2, 3])
        loss_ac = ACLoss_func(logit[:, :1]).mean(dim=[0, 2, 3]) # [1]
        # [|C0:t-1|]
        loss_dkd_pos = KDLoss_func(
            features['pos_reg'][:, :n_old_classes],
            features_old['pos_reg'].sigmoid()
        ).mean(dim=[0, 2, 3])
        # [|C0:t-1|]
        loss_dkd_neg = KDLoss_func(
            features['neg_reg'][:, :n_old_classes],
            features_old['neg_reg'].sigmoid()
        ).mean(dim=[0, 2, 3])
    else :
        raise NotImplementedError
    return loss_mbce, loss_kd, loss_ac, loss_dkd_pos, loss_dkd_neg

def loss_MiB(logit, label, n_old_classes, n_new_classes,
            UnCELoss_func, KDLoss_func = None, logit_old = None):
    if KDLoss_func is None:
        assert logit_old is None, \
            "KDLoss_func is None but logit_old is not None"
    
    loss_CE = UnCELoss_func(logit, label).mean()#dim=[0, 2, 3])
    if KDLoss_func is None:
        # step 0 loss
        loss_KD = 0
    else:
        loss_KD = KDLoss_func(logit,logit_old).mean()#dim=[0, 2, 3])
    return loss_CE, loss_KD