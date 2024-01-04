import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from models.loss import WBCELoss, KDLoss, ACLoss, UnbiasedCrossEntropy, UnbiasedKnowledgeDistillationLoss, features_distillation
from data_loader import VOC
from data_loader import ADE
from models.loss_method import loss_DKD, loss_MiB, loss_PLOP
import wandb
import numpy as np
from utils import entropy
import gc

class Trainer_base(BaseTrainer):
    """
    Trainer class for a base step
    """
    def __init__(
        self, model, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False

            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.n_old_classes = len(self.task_info['old_class'])  # 0
        # voc : 19-1: 19 | 15-5: 15 | 15-1: 15...
        # ade : 100-50: 100 | 100-10: 100 | 50-50: 50 |
        self.n_new_classes = len(self.task_info['new_class'])  
        # self.name = self.config['name']
        # self.method = self.config['method']
        self.step = self.task_info['step']
        self.dataset_type = self.task_info['dataset']

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        if self.config['method'] == 'DKD':
            self.loss_name = ['loss', 'loss_mbce', 'loss_ac']
        elif self.config['method'] == 'MiB':
            self.loss_name = ['loss', 'loss_CE', 'loss_KD']
        elif self.config['method'] == 'PLOP':
            self.loss_name = ['loss', 'loss_CE', 'loss_POD']
        elif self.config['method'] == 'base':
            self.loss_name = ['loss', 'loss_mbce']
        else :
            raise NotImplementedError(self.config['method'])
        self.train_metrics = MetricTracker(
            keys=self.loss_name,
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])
        if self.config['method'] == 'DKD':
            pos_weight = torch.ones([len(self.task_info['new_class'])], device=self.device) * self.config['hyperparameter']['pos_weight']
            self.BCELoss = WBCELoss(pos_weight=pos_weight, n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
            self.ACLoss = ACLoss()
        elif self.config['method'] == 'MiB' :
            self.CEloss = UnbiasedCrossEntropy(old_cl=self.n_old_classes, ignore_index=255, reduction='none')
        elif self.config['method'] == 'PLOP':
            self.CEloss = nn.CrossEntropyLoss(ignore_index=255,reduction='none')
        elif self.config['method'] == 'base':
            pos_weight = torch.ones([len(self.task_info['new_class'])], device=self.device) * self.config['hyperparameter']['pos_weight']
            self.BCELoss = WBCELoss(pos_weight=None, n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        else :
            raise NotImplementedError(self.config['method'])

        self._print_train_info()

    def _print_train_info(self):
        if self.config['method'] == 'DKD':
            self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
            self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['ac']} * L_ac")
        elif self.config['method'] == 'MiB' :
            self.logger.info(f"Total loss = L_UnCE + {self.config['hyperparameter']['kd']}")
        elif self.config['method'] == 'PLOP':
            self.logger.info(f"Total loss = L_CE")
        elif self.config['method'] == 'base':
            self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
            self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce")
        else :
            raise NotImplementedError(self.config['method'])
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
        else:
            self.model.freeze_bn(affine_freeze=False)

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            # print(data['image'].shape, data['label'].shape) # torch.Size([6, 3, 512, 512]) torch.Size([6, 512, 512])
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, features = self.model(data['image'], ret_intermediate=False)
                if self.config['method'] == 'DKD':    
                    loss_mbce, _, loss_ac, _, _ = loss_DKD(logit, data['label'], 
                                                           self.n_old_classes, self.n_new_classes,\
                                                            self.BCELoss, self.ACLoss)
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + self.config['hyperparameter']['ac'] * loss_ac.sum()
                elif self.config['method'] == 'MiB':
                    loss_CE, _ = loss_MiB(logit, data['label'], 
                                          self.n_old_classes, self.n_new_classes,
                                            self.CEloss)
                    loss = loss_CE
                elif self.config['method'] == 'PLOP':
                    loss_CE, _ = loss_PLOP(logit, data['label'],
                                     self.n_old_classes, self.n_new_classes, 
                                     self.CEloss)
                    loss = loss_CE
                elif self.config['method'] == 'base':
                    loss_mbce = self.BCELoss(logit[:,-self.n_new_classes:], data['label']).mean(dim=[0, 2, 3])
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum()
                else:
                    raise NotImplementedError(self.config['method'])
                
                gc.collect()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            if self.config['method'] == 'DKD':   
                self.train_metrics.update('loss_mbce', loss_mbce.mean().item())
                self.train_metrics.update('loss_ac', loss_ac.mean().item())
            elif self.config['method'] == 'MiB' or self.config['method'] == 'PLOP':
                self.train_metrics.update('loss_CE', loss_CE.mean().item())
            elif self.config['method'] == 'base':
                self.train_metrics.update('loss_mbce', loss_mbce.mean().item())
            else :
                raise NotImplementedError(self.config['method'])
            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()
        
        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")
        wandb_log_done = False
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _ = self.model(data['image'])
                if self.config['method'] == 'DKD':
                    logit = torch.sigmoid(logit)
                    pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                    idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                    idx = idx.sum(dim=1)  # logit: [N, H, W]
                    pred[idx == 0] = 0  # set background (non-target class)
                elif self.config['method'] == 'MiB' or self.config['method'] == 'PLOP':
                    _, pred = logit.max(dim=1)
                elif self.config['method'] == 'base':
                    _, pred = logit.max(dim=1)
                else :
                    raise NotImplementedError(self.config['method'])
                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)
                labels = data['label'].type(torch.long)
                if (batch_idx == len(self.test_loader) -1 and wandb_log_done is False) or \
                    wandb_log_done is False and ( any([lbl in torch.unique(labels) for lbl in self.task_info['new_class']]) ) :
                    
                    img = (self.denorm(data['image'][0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1,2,0)
                    pred = self.label2color(pred[0]).astype(np.uint8)
                    label = self.label2color(labels[0].detach().cpu().numpy()).astype(np.uint8)
                    concat_img = np.concatenate((img, pred, label), axis=1)  # concat along width, then make H,W,C

                    self.logger.log_wandb({'val/image' : [wandb.Image(concat_img, caption=f'input,pred,label')]},step=epoch)
                    wandb_log_done = True

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})

                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys() and self.dataset_type == 'voc':
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_val.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{VOC[i]} {met()['by_class'][i]:.2f}\n"
                        elif i in self.evaluator_val.old_classes_idx:
                            by_class_str = by_class_str + f"{i:2d}  {VOC[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
            # log.update({met.__name__ + '_confusion_matrix': met().confusion_matrix})
        wandb_log = {}
        for key, value in log.items():
            if 'by_class' not in key:
                wandb_log.update({f"val/{key}": float(value)})
            # if 'confusion_matrix' in key: # TODO : not ready
            #     wandb_log.update({f"val/{key}": wandb.plot.confusion_matrix(probs=None,
            #                                                                 y_true=value.sum(axis=1),
            #                                                                 preds=value.sum(axis=0),
            #                                                                 class_names=VOC)})
            else :
                by_class = []
                for s in value.split("\n"):
                    if s == '':
                        continue
                    idx, name, val = [i for i in s.split(" ") if i != '']
                    by_class.append([int(idx), name, float(val)])
                wandb_log.update({f"val/{key}": wandb.Table(data=by_class, columns=["idx", "name", "value"])})
                    
        self.logger.log_wandb(wandb_log,step=epoch)
        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        self.model.eval()
        wandb_log_done = False
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, features = self.model(data['image'])
                if self.config['method'] == 'DKD':
                    logit = torch.sigmoid(logit)
                    pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]

                    idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                    idx = idx.sum(dim=1)  # logit: [N, H, W]

                    pred[idx == 0] = 0  # set background (non-target class)
                # elif 'MIB' or 'PLOP'
                elif self.config['method'] == 'MiB' or self.config['method'] == 'PLOP':
                    _, pred = logit.max(dim=1)
                elif self.config['method'] == 'base':
                    _, pred = logit.max(dim=1)
                else:
                    raise NotImplementedError(self.config['method'])
                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

                labels = data['label'].type(torch.long)
                if (batch_idx == len(self.test_loader) -1 and wandb_log_done is False) or \
                    wandb_log_done is False and ( any([lbl in torch.unique(labels) for lbl in self.task_info['new_class']]) ) :
                    
                    img = (self.denorm(data['image'][0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1,2,0)
                    pred = self.label2color(pred[0]).astype(np.uint8)
                    label = self.label2color(labels[0].detach().cpu().numpy()).astype(np.uint8)
                    concat_img = np.concatenate((img, pred, label), axis=1)  # concat along width, then make H,W,C

                    self.logger.log_wandb({'test/image' : [wandb.Image(concat_img, caption=f'input,pred,label')]},step=epoch)
                    wandb_log_done = True
            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                    
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys() and self.dataset_type == 'voc':
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{VOC[i]} {met()['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {VOC[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        
        wandb_log = {}
        for key, value in log.items():
            if 'by_class' not in key:
                wandb_log.update({f"test/{key}": float(value)})
            else :
                by_class = []
                for s in value.split("\n"):
                    if s == '':
                        continue
                    idx, name, val = [i for i in s.split(" ") if i != '']
                    by_class.append([int(idx), name, float(val)])
                wandb_log.update({f"test/{key}": wandb.Table(data=by_class, columns=["idx", "name", "value"])})
                    
        self.logger.log_wandb(wandb_log,step=epoch)
        return log


class Trainer_incremental(Trainer_base):
    """
    Trainer class for incremental steps
    """
    def __init__(
        self, model, model_old, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(
            model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
            data_loader=data_loader, lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)
        self.threshold = 0.001 #entropy threshold
        self.pseudo_labeling = 'entropy'
        self.n_current_classes =  self.n_old_classes + self.n_new_classes
        self.nb_new_classes = self.n_new_classes
        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)

        if self.config['method'] == 'DKD':
            self.loss_name = ['loss', 'loss_mbce', 'loss_kd', 'loss_ac', 'loss_dkd_pos', 'loss_dkd_neg']
        elif self.config['method'] == 'MiB':
            self.loss_name = ['loss', 'loss_CE', 'loss_KD']
        elif self.config['method'] == 'PLOP':
            self.loss_name = ['loss', 'loss_CE', 'loss_POD']
        elif self.config['method'] == 'base':
            self.loss_name = ['loss', 'loss_mbce']
        else :
            raise NotImplementedError(self.config['method'])
        self.train_metrics = MetricTracker(
            keys=self.loss_name,
            writer=self.writer, colums=['total', 'counts', 'average'],
        )
        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        if self.config['method'] == 'DKD':
            self.KDLoss = KDLoss(pos_weight=None, reduction='none')
        elif self.config['method'] == 'MiB' :
            self.KDLoss = UnbiasedKnowledgeDistillationLoss(alpha=self.config['hyperparameter']['alpha'])
        elif self.config['method'] == 'PLOP':
            self.PodLoss = features_distillation
        elif self.config['method'] == 'base':
            pass
        else :
            raise NotImplementedError(self.config['method'])

    def _print_train_info(self):        
        if self.config['method'] == 'DKD':
            self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
            self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['kd']} * L_kd "
                            f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos + {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                            f"+ {self.config['hyperparameter']['ac']} * L_ac")
        elif self.config['method'] == 'MiB' :
            self.logger.info(f"Total loss = L_UnCE + {self.config['hyperparameter']['kd']} * L_UnKD (alpha : {self.config['hyperparameter']['alpha']})")
        elif self.config['method'] == 'PLOP':
            self.logger.info(f"Total loss = L_CE + POD_loss")
        elif self.config['method'] == 'base':
            self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
            self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce")
        else :
            raise NotImplementedError(self.config['method'])
    def _before_train(self):
        if self.config['method'] == 'PLOP':
            if self.pseudo_labeling is None:
                return
            if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:
                self.logger.info("Find median score")
                self.thresholds, _ = self.find_best_value(mode="median")
            elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0: # in plop
                self.logger.info("Find median score with entropy")
                self.thresholds, self.max_entropy = self.find_best_value(mode="entropy")
        else :
            self.logger.info(f"No jobs before train {self.config['method']}")
            return
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
            self.model.module.freeze_dropout()
        else:
            self.model.freeze_bn(affine_freeze=False)
            self.model.freeze_dropout()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        wandb_log_done = False
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            labels = data['label'].type(torch.long)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, features = self.model(data['image'], ret_intermediate=True)

                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old = self.model_old(data['image'], ret_intermediate=True)
                if self.config['method'] == 'DKD':
                    loss_out = loss_DKD(logit, data['label'],self.n_old_classes, self.n_new_classes, \
                                        self.BCELoss, self.ACLoss, self.KDLoss, logit_old, features, features_old)
                    loss_mbce, loss_kd, loss_ac, loss_dkd_pos, loss_dkd_neg = loss_out
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                        self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                        self.config['hyperparameter']['ac'] * loss_ac.sum()
                elif self.config['method'] == 'MiB':
                    loss_out = loss_MiB(logit, data['label'], self.n_old_classes, self.n_new_classes,
                                                self.CEloss,self.KDLoss, logit_old)
                    loss_CE, loss_KD = loss_out
                    loss = loss_CE + self.config['hyperparameter']['kd'] * loss_KD
                elif self.config['method'] == 'PLOP':
                    
                    ############
                    # psuedo labeling
                    # original_labels = labels.clone()
                    mask_background = labels < self.n_old_classes
                    # self.pseudo_labeling == "entropy":
                    probs = torch.softmax(logit_old, dim=1)
                    # max_probs, pseudo_labels = probs.max(dim=1)
                    _, pseudo_labels = probs.max(dim=1)
                    mask_valid_pseudo = (entropy(probs) /
                                         self.max_entropy) < self.thresholds[pseudo_labels]
                    ############
                    # All old labels that are NOT confident enough to be used as pseudo labels:
                    labels[~mask_valid_pseudo & mask_background] = 255
                    labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                        mask_background]
                    del probs, pseudo_labels
                    ############
                    # Number of old/bg pixels that are certain
                    num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
                    # Number of old/bg pixels
                    den =  mask_background.float().sum(dim=(1,2))
                    # If all old/bg pixels are certain the factor is 1 (loss not changed)
                    # Else the factor is < 1, i.e. the loss is reduced to avoid
                    # giving too much importance to new pixels
                    classif_adaptive_factor = num / (den + 1e-6)
                    classif_adaptive_factor = classif_adaptive_factor[:, None, None]
                    del num, den
                    # pos_neg
                    attentions_old = features_old["attentions"]
                    attentions = features["attentions"]
                    # if self.pod_logits: True
                    attentions_old.append(features_old["sem_logits_small"])
                    attentions.append(features["sem_logits_small"])
                    loss_out = loss_PLOP(logit, labels,
                                         self.n_old_classes, self.n_new_classes, 
                                         self.CEloss, self.PodLoss, logit_old, 
                                         attentions, attentions_old,
                                         classif_adaptive_factor)
                    loss_CE, loss_POD = loss_out
                    loss = loss_CE + loss_POD
                elif self.config['method'] == 'base':
                    loss_mbce = self.BCELoss(logit[:,-self.n_new_classes:], data['label']).mean(dim=[0, 2, 3])
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum()
                else :
                    raise NotImplementedError(self.config['method'])
                
                if (batch_idx == len(self.train_loader) -1 and wandb_log_done is False) or \
                    wandb_log_done is False and ( any([lbl in torch.unique(labels) for lbl in self.task_info['new_class']]) ) :
                    loss_dict = {'loss': loss.item()}
                    for key in self.loss_name[1:]:
                        loss_dict.update({key: loss_out[self.loss_name.index(key)-1].mean().item()})
                    self.logger.log_wandb(loss_dict,step=epoch)
                    _, pred = logit.max(dim=1)
                    _, pred_old = logit_old.max(dim=1)
                    img = (self.denorm(data['image'][0].detach().cpu().numpy()) * 255).astype(np.uint8)
                    pred = self.label2color(pred[0].detach().cpu().numpy()).transpose(2, 0, 1).astype(np.uint8)
                    pred_old = self.label2color(pred_old[0].detach().cpu().numpy()).transpose(2, 0, 1).astype(np.uint8)
                    label = self.label2color(labels[0].detach().cpu().numpy()).transpose(2, 0, 1).astype(np.uint8)
                    concat_img = np.concatenate((img, pred,pred_old, label), axis=2).transpose(1,2,0)  # concat along width, then make H,W,C

                    self.logger.log_wandb({'train/image' : [wandb.Image(concat_img, caption=f'input,pred,pred_old,label')]},step=epoch)
                    del concat_img, img, pred, pred_old, label
                    wandb_log_done = True
                gc.collect()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            if self.config['method'] == 'DKD':
                self.train_metrics.update('loss_mbce', loss_mbce.mean().item())
                self.train_metrics.update('loss_kd', loss_kd.mean().item())
                self.train_metrics.update('loss_ac', loss_ac.mean().item())
                self.train_metrics.update('loss_dkd_pos', loss_dkd_pos.mean().item())
                self.train_metrics.update('loss_dkd_neg', loss_dkd_neg.mean().item())
            elif self.config['method'] == 'MiB':
                self.train_metrics.update('loss_CE', loss_CE.mean().item())
                self.train_metrics.update('loss_KD', loss_KD.mean().item())
            elif self.config['method'] == 'PLOP':
                self.train_metrics.update('loss_CE', loss_CE.mean().item())
                self.train_metrics.update('loss_POD', loss_POD.mean().item())
            elif self.config['method'] == 'base':
                self.train_metrics.update('loss_mbce', loss_mbce.mean().item())
            else :
                raise NotImplementedError(self.config['method'])
            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def find_best_value(self,mode="probability"):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.n_current_classes).float().to(self.device))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]

        histograms = torch.zeros(self.n_current_classes, nb_bins).long().to(self.device)

        for cur_step, data in enumerate(self.train_loader):
            images, labels = data['image'], data['label']
            del data
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            outputs_old, features_old = self.model_old(images, ret_intermediate=False)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True
            )

            if cur_step % 10 == 0:
                self.logger.info(f"Median computing {cur_step}/{len(self.train_loader)}.")
            import gc
            gc.collect()
        thresholds = torch.zeros(self.n_current_classes, dtype=torch.float32).to(
            self.device
        )  # zeros or ones? If old_model never predict a class it may be important

        self.logger.info("Approximating median")
        for c in range(self.n_current_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)
        # if self.step_threshold is not None:
        #     self.threshold += self.step * self.step_threshold

        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        self.logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(self.device), max_value
