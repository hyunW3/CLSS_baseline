import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# from psuedo_labeling import PLOP_pseudo_labeling

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

def loss_PLOP(logits, labels, n_old_classes, n_new_classes,
              CE_Loss_func, PodLoss_func=None, logits_old=None ,
              attentions=None, attentions_old=None,classif_adaptive_factor=None):
    if PodLoss_func is None:
        assert attentions is None and logits_old is None, \
            "PodLoss_func is not None or logits_old is not None"
    labels = labels.long()
    if attentions is None and PodLoss_func is None:
        # step 0 loss
        loss_ce = CE_Loss_func(
                    logits,  # [N, |C0:t-1|], H, W]
                    labels,                # [N, H, W]
                ).mean() 
        loss_pod = 0
    else:
        # step 1 ~ loss
        # [|Ct|]
        loss_ce = CE_Loss_func(
                    logits,  # [N, |C0:t-1|], H, W]
                    labels,                # [N, H, W]
                ) # B,H,W
        loss_ce = (classif_adaptive_factor * loss_ce).mean() 
        print(f"loss_ce : {loss_ce}, {loss_ce.shape}")
        # [|C0:t-1|]
        # TODO : need to check
        n_current_classes = n_old_classes + n_new_classes # the number of current classes (old + new)
        loss_pod = PodLoss_func(
            attentions,
            attentions_old,
            index = label,
            index_new_class=n_old_classes+1, # ? # 15-1 step1 => 16
            outputs_old=logits_old,
            nb_current_classes=n_current_classes,
            nb_new_classes = n_new_classes
        )

    return loss_ce, loss_pod
if __name__ == "__main__":
    load = True
    save = not load
    if load is False:
        logits = torch.randn(2, 18, 10, 10)
        label = torch.randint(0, 18, (2, 10, 10))
        n_old_classes = 14
        n_new_classes = 3
        logits_old = torch.randn(2, n_old_classes+1, 10, 10)
    else :
        data_path = "./data"
        logits = torch.load(f"{data_path}/logits.pt")
        label = torch.load(f"{data_path}/label.pt")
        logits_old = torch.load(f"{data_path}/logits_old.pt")
        config = torch.load(f"{data_path}/config.pt")
        n_old_classes = config["n_old_classes"]
        n_new_classes = config["n_new_classes"]
        alpha = config["alpha"]
    # print(logits.shape,label.shape, logits_old.shape)
    if save is True:
        data_path = "./data"
        import os
        os.makedirs(data_path, exist_ok=True)
        torch.save(logits, f"{data_path}/logits.pt")
        torch.save(label, f"{data_path}/label.pt")
        torch.save(logits_old, f"{data_path}/logits_old.pt")
        torch.save( {"n_old_classes":n_old_classes, 
                                    "n_new_classes" : n_new_classes,
                                    "alpha" : 1.0},f"{data_path}/config.pt")
    from loss import UnbiasedCrossEntropy, UnbiasedKnowledgeDistillationLoss
    CELoss = UnbiasedCrossEntropy(old_cl=n_old_classes, ignore_index=255, reduction='none')
    KDLoss = UnbiasedKnowledgeDistillationLoss(alpha=1.0)
    loss_CE, loss_KD = loss_MiB(logits, label, n_old_classes, n_new_classes, CELoss, KDLoss, logits_old)
    print(f"loss_CE: {loss_CE:.6f} loss_KD : {loss_KD:.6f}" )