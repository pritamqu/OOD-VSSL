# this script is taken from https://github.com/facebookresearch/AVID-CMA

import datetime
import sys
import torch
from torch import distributed as dist
from collections import deque
import numpy as np
from scipy import stats
from sklearn import metrics
import time

class Logger(object):
    def __init__(self, quiet=False, log_fn=None, rank=0, prefix=""):
        self.rank = rank if rank is not None else 0
        self.quiet = quiet
        self.log_fn = log_fn

        self.prefix = ""
        if prefix:
            self.prefix = prefix + ' | '

        self.file_pointers = []
        if self.rank == 0:
            if self.quiet:
                open(log_fn, 'a').close() # change to w --> a as to append during resume mode

    def add_line(self, content):
        if self.rank == 0:
            msg = self.prefix+content
            if self.quiet:
                fp = open(self.log_fn, 'a')
                fp.write(msg+'\n')
                fp.flush()
                fp.close()
            else:
                print(msg)
                sys.stdout.flush()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, phase, epoch=None, logger=None, tb_writter=None):
        self.batches_per_epoch = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(epoch, num_batches)
        self.meters = meters
        self.phase = phase
        self.epoch = epoch
        self.logger = logger
        self.tb_writter = tb_writter

    def display(self, batch):
        step = self.epoch * self.batches_per_epoch + batch
        date = str(datetime.datetime.now())
        entries = ['{} | {} {}'.format(date, self.phase, self.batch_fmtstr.format(batch))]
        entries += [str(meter) for meter in self.meters]
        if self.logger is None:
            print('\t'.join(entries))
        else:
            self.logger.add_line('\t'.join(entries))

        if self.tb_writter is not None:
            for meter in self.meters:
                self.tb_writter.add_scalar('{}/{}'.format(self.phase, meter.name), meter.val, step)

    def _get_batch_fmtstr(self, epoch, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        epoch_str = '[{}]'.format(epoch) if epoch is not None else ''
        return epoch_str+'[' + fmt + '/' + fmt.format(num_batches) + ']'

    def synchronize_meters(self, cur_gpu):
        metrics = torch.tensor([m.avg for m in self.meters]).cuda(cur_gpu)
        metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(metrics_gather, metrics)

        metrics = torch.stack(metrics_gather).float().mean(0).cpu().numpy()
        for meter, m in zip(self.meters, metrics):
            meter.avg = m
            
    def synchronize_meters_custom(self, cur_gpu):
        # slightly modifying some stuffs related to time, show total time than avg
        metrics = torch.tensor([m.sum if m.name in ['Time', 'Data'] else m.avg for m in self.meters]).cuda(cur_gpu)
        metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(metrics_gather, metrics)

        metrics = torch.stack(metrics_gather).float().mean(0).cpu().numpy()
        for meter, m in zip(self.meters, metrics):
            if meter.name in ['Time', 'Data']:
                meter.sum = m
            else:
                meter.avg = m

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', window_size=0):
        self.name = name
        self.fmt = fmt
        self.window_size = window_size
        self.reset()

    def reset(self):
        if self.window_size > 0:
            self.q = deque(maxlen=self.window_size)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if self.window_size > 0:
            self.q.append((val, n))
            self.count = sum([n for v, n in self.q])
            self.sum = sum([v * n for v, n in self.q])
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
  

def synchronize_holder(holder, cur_gpu):
    metrics = torch.tensor(holder)#.cuda(cur_gpu)
    metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
    dist.all_gather(metrics_gather, metrics)
    return metrics_gather

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def mean_ap_metric(predicts, targets, logger=None):
    # copied from https://github.com/facebookresearch/video-long-term-feature-banks/blob/main/lib/utils/metrics.py
    """Compute mAP, wAP, AUC for Charades."""

    # predicts = np.vstack(predicts)
    # targets = np.vstack(targets)
    if logger is not None:
        logger.add_line(f"Getting mAP for {predicts.shape[0]} examples")
    start_time = time.time()

    
    predict = predicts[:, ~np.all(targets == 0, axis=0)]
    target = targets[:, ~np.all(targets == 0, axis=0)]
    mean_auc = 0
    aps = [0]
    try:
        mean_auc = metrics.roc_auc_score(target, predict)
    except ValueError:
        print(
            'The roc_auc curve requires a sufficient number of classes \
            which are missing in this sample.'
        )
    try:
        aps = metrics.average_precision_score(target, predict, average=None)
    except ValueError:
        print(
            'Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample.'
        )

    mean_ap = np.mean(aps)
    weights = np.sum(target.astype(float), axis=0)
    weights /= np.sum(weights)
    mean_wap = np.sum(np.multiply(aps, weights))
    all_aps = np.zeros((1, targets.shape[1]))
    all_aps[:, ~np.all(targets == 0, axis=0)] = aps
    if logger is not None:
        logger.add_line(f'Done in {time.time() - start_time} seconds')
        logger.add_line(f'mean_auc: {mean_auc} - mean_ap: {mean_ap} - mean_wap: {mean_wap}')
    
    
    return {'mean_auc': mean_auc, 
            'mean_ap': mean_ap, 
            'mean_wap': mean_wap, 
            'all_aps': all_aps.flatten()}


def calculate_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res[f'top{k}'] = correct_k.mul_(100.0 / batch_size)
        return res
    
def calculate_meanAP(predicts, targets):
    # copied from https://github.com/facebookresearch/video-long-term-feature-banks/blob/main/lib/utils/metrics.py
    """Compute mAP, wAP, AUC for Charades."""

    # # predicts = np.vstack(predicts)
    # # targets = np.vstack(targets)
    # if logger is not None:
    #     logger.add_line(f"Getting mAP for {predicts.shape[0]} examples")
    # # start_time = time.time()

    predict = predicts[:, ~np.all(targets == 0, axis=0)]
    target = targets[:, ~np.all(targets == 0, axis=0)]
    mean_auc = 0
    aps = [0]
    try:
        mean_auc = metrics.roc_auc_score(target, predict)
    except ValueError:
        print(
            'The roc_auc curve requires a sufficient number of classes \
            which are missing in this sample.'
        )
    try:
        aps = metrics.average_precision_score(target, predict, average=None)
    except ValueError:
        print(
            'Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample.'
        )

    mean_ap = np.mean(aps)
    weights = np.sum(target.astype(float), axis=0)
    weights /= np.sum(weights)
    mean_wap = np.sum(np.multiply(aps, weights))
    all_aps = np.zeros((1, targets.shape[1]))
    all_aps[:, ~np.all(targets == 0, axis=0)] = aps
    # if logger is not None:
    #     logger.add_line('\tDone in {time.time() - start_time} seconds')
    
    return {'mean_auc': mean_auc, 
            'mean_ap': mean_ap, 
            'mean_wap': mean_wap, 
            # 'all_aps': all_aps.flatten()
            }
    
def any_in_list(list1, list2):
    return bool(set(list1) & set(list2))

# def calculate_prec_recall_f1(output, target, 
#                               ):
#     # work for single class classification
#     # need to see if this works with multi-label
#     results_actions = metrics.precision_recall_fscore_support(target.cpu().numpy(), torch.argmax(output, dim=-1).cpu().numpy(), average=None)
#     f1_scores, precision, recall = results_actions[2], results_actions[0], results_actions[1]
#     f1_mean, prec_mean, rec_mean = np.mean(f1_scores), np.mean(precision), np.mean(recall)
#    
#     return {'f1': f1_mean,
#             'precision': prec_mean,
#             'recall': rec_mean,
#             }

def calculate_prec_recall_f1(output, ground_truth, threshold=0.5,
                              ):
    
    predictions = (output.cpu().numpy() > threshold).astype(int)
    ground_truth = (ground_truth.cpu().numpy() > threshold).astype(int)

    # Scikit f1 score implementation
    prec_a, rec_a, f1_a, _ = metrics.precision_recall_fscore_support(ground_truth, predictions, average='samples')
    acc_a = metrics.accuracy_score(ground_truth, predictions)
    
    return {'f1': f1_a,
            'precision': prec_a,
            'recall': rec_a,
            'acc': acc_a}
    
def all_evals(output, target, eval_metrics):
    results = {}

    if any_in_list(eval_metrics, ['top1', 'top5']):
        if 'top5' in eval_metrics:
            _res = calculate_topk(output, target, topk=(1,5))
        else:
            _res = calculate_topk(output, target, topk=(1,))
        for _r in _res:
            results[_r] = _res[_r].item()
    
    if any_in_list(eval_metrics, ['mean_auc', 'mean_ap', 'mean_wap']):
        _res = calculate_meanAP(output, target)
        for _r in _res:
            results[_r] = _res[_r]
            
    if any_in_list(eval_metrics, ['f1', 'precision', 'recall', 'acc']):
        _res = calculate_prec_recall_f1(output, target)
        for _r in _res:
            results[_r] = _res[_r]
            
    return results
    

