
# src: https://github.com/Cogito2012/DEAR/blob/master/mmaction/models/heads/debias_head.py


import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import xlogy
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tools import synchronize_holder

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def update_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def run_inference(model, data_loader, args, cfg, amp, logger, fwd_kwargs):
    
    # run on k400 test

    evidence = cfg['evidence']
    uncertainty = cfg['uncertainty']
    
    model.eval()
    if not cfg['uncertainty'] == 'EDL':
        raise NotImplementedError()
        # need to make it dist ready
        # all_confidences, all_uncertainties, all_results, all_gts = run_stochastic_inference(model, data_loader, uncertainty, args, amp, fwd_kwargs)
    else:
        all_confidences, all_uncertainties, all_results, all_gts = \
            run_evidence_inference_dist(model, data_loader, evidence, args, amp, logger, fwd_kwargs, 
                                        average_clips=cfg['average_clips'])
    return all_confidences, all_uncertainties, all_results, all_gts


def evidence_to_prob(output, evidence_type):
    if evidence_type == 'relu':
        get_evidence = relu_evidence 
    elif evidence_type == 'exp':
        get_evidence = exp_evidence
    elif evidence_type == 'softplus':
        get_evidence = softplus_evidence
    else:
        raise NotImplementedError
       
    alpha = get_evidence(output) + 1
    S = torch.sum(alpha, dim=-1, keepdim=True)
    prob = alpha / S
    return prob
    
def run_evidence_inference(model, data_loader, evidence_type, args, amp, fwd_kwargs, average_clips='evidence'):

    # get the evidence function
    if evidence_type == 'relu':
        get_evidence = relu_evidence 
    elif evidence_type == 'exp':
        get_evidence = exp_evidence
    elif evidence_type == 'softplus':
        get_evidence = softplus_evidence
    else:
        raise NotImplementedError
        
    num_classes = model.module.classifier.num_classes
    is_video = True if data_loader.dataset.mode == 'video' else False
    
    # model = MMDataParallel(model, device_ids=[0])
    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    # prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, sample in enumerate(data_loader):
        with torch.no_grad():
            video = sample['frames']
            target = sample['label'].cuda()
            video = video.cuda(args.gpu, non_blocking=True)
            
            # tackle video
            if is_video: # test_dense
                batch_size, clips_per_sample = video.shape[0], video.shape[1]
                video = video.flatten(0, 1).contiguous()
                
            if amp is not None:
                with torch.cuda.amp.autocast():
                    output = model.module.forward_test(video, target, **fwd_kwargs)
            else:
                raise NotImplementedError()
                
            if is_video:
                output = output.view(batch_size, clips_per_sample, -1)
                if average_clips == 'prob':
                    output = F.softmax(output, dim=2).mean(dim=1)
                elif average_clips == 'score':
                    output = output.mean(dim=1)
                elif average_clips == 'evidence':
                    # assert 'evidence_type' in self.test_cfg.keys()
                    output = evidence_to_prob(output, evidence_type)
                    output = output.mean(dim=1)
                
                # output = output.view(batch_size, clips_per_sample, -1).mean(1) # mean of all clips
            
            # output = model(return_loss=False, **data)
            evidence = get_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1)
            scores = alpha / torch.sum(alpha, dim=1, keepdim=True)
            scores = scores.cpu()
            uncertainty = uncertainty.cpu()
            
        all_uncertainties.append(uncertainty.numpy())
        # compute the predictions and save labels
        preds = np.argmax(scores.numpy(), axis=1)
        all_results.append(preds)
        conf = np.max(scores.numpy(), axis=1)
        all_confidences.append(conf)
        
        labels = sample['label'].numpy()
        all_gts.append(labels)

        # # use the first key as main key to calculate the batch size
        # batch_size = len(next(iter(data.values())))
        # for _ in range(batch_size):
        #     prog_bar.update()
    all_confidences = np.concatenate(all_confidences, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_results = np.concatenate(all_results, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    return all_confidences, all_uncertainties, all_results, all_gts


def run_evidence_inference_dist(model, data_loader, evidence_type, args, amp, logger, fwd_kwargs, average_clips='score'):

    # get the evidence function
    if evidence_type == 'relu':
        get_evidence = relu_evidence 
    elif evidence_type == 'exp':
        get_evidence = exp_evidence
    elif evidence_type == 'softplus':
        get_evidence = softplus_evidence
    else:
        raise NotImplementedError
        
    num_classes = model.module.classifier.num_classes
    is_video = True if data_loader.dataset.mode == 'video' else False
    
    # model = MMDataParallel(model, device_ids=[0])
    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    # prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, sample in enumerate(data_loader):
        with torch.no_grad():
            video = sample['frames']
            target = sample['label'].cuda()
            video = video.cuda(args.gpu, non_blocking=True)
            
            # tackle video
            if is_video: # test_dense
                batch_size, clips_per_sample = video.shape[0], video.shape[1]
                video = video.flatten(0, 1).contiguous()
                
            if amp is not None:
                with torch.cuda.amp.autocast():
                    output = model.module.forward_test(video, target, **fwd_kwargs)
            else:
                raise NotImplementedError()
                
            if is_video:
                output = output.view(batch_size, clips_per_sample, -1)
                if average_clips == 'prob':
                    output = F.softmax(output, dim=2).mean(dim=1)
                elif average_clips == 'score':
                    output = output.mean(dim=1)
                elif average_clips == 'evidence': # this seems have some problem
                    # TODO: figure out why this is causing problem, the original paper uses that.
                    # assert 'evidence_type' in self.test_cfg.keys()
                    output = evidence_to_prob(output, evidence_type)
                    output = output.mean(dim=1)
                
                # output = output.view(batch_size, clips_per_sample, -1).mean(1) # mean of all clips
            
            if len(output.shape)==1:
                output = output.unsqueeze(0)
            evidence = get_evidence(output)
            alpha = evidence + 1
            
            uncertainty = num_classes / torch.sum(alpha, dim=1)
            scores = alpha / torch.sum(alpha, dim=1, keepdim=True)
            # scores = scores.cpu()
            # uncertainty = uncertainty.cpu()
            
        all_uncertainties.append(uncertainty.unsqueeze(-1))
        # compute the predictions and save labels
        preds = torch.argmax(scores, axis=1)
        all_results.append(preds.unsqueeze(-1))
        conf = torch.max(scores, axis=1)[0]
        all_confidences.append(conf.unsqueeze(-1))
        # labels = sample['label']
        all_gts.append(target.unsqueeze(-1))

        # if i+1 % 100==0:
        logger.add_line(f"Iter: {i}/{len(data_loader)}")
        # # use the first key as main key to calculate the batch size
        # batch_size = len(next(iter(data.values())))
        # for _ in range(batch_size):
        #     prog_bar.update()
    # all_confidences = np.concatenate(all_confidences, axis=0)
    # all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    # all_results = np.concatenate(all_results, axis=0)
    # all_gts = np.concatenate(all_gts, axis=0)
    
    all_uncertainties = torch.vstack(all_uncertainties)
    all_results = torch.vstack(all_results)
    all_confidences = torch.vstack(all_confidences)
    all_gts = torch.vstack(all_gts)
    
    if args.distributed:
        all_uncertainties = synchronize_holder(all_uncertainties, args.gpu)
        all_results = synchronize_holder(all_results, args.gpu)
        all_confidences = synchronize_holder(all_confidences, args.gpu)
        all_gts = synchronize_holder(all_gts, args.gpu)
    else:
        all_uncertainties = [all_uncertainties]
        all_results = [all_results]
        all_confidences = [all_confidences]
        all_gts = [all_gts]

    all_uncertainties = torch.cat(all_uncertainties)
    all_results = torch.cat(all_results) 
    all_confidences = torch.cat(all_confidences) 
    all_gts = torch.cat(all_gts) 

    

    return all_confidences.squeeze().cpu().numpy(), \
        all_uncertainties.squeeze().cpu().numpy(), \
            all_results.squeeze().cpu().numpy(), \
                all_gts.squeeze().cpu().numpy()

    
def run_stochastic_inference(model, data_loader, uncertainty, args, amp, fwd_kwargs, npass=10):
    # model = MMDataParallel(model, device_ids=[0])
    all_confidences, all_uncertainties, all_results, all_gts = [], [], [], []
    # prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    softmax = torch.nn.Softmax(dim=1)
    is_video = True if data_loader.dataset.mode == 'video' else False
    
    
    for i, sample in enumerate(data_loader):
        all_scores = []
        with torch.no_grad():
            for n in range(npass):
                # set new random seed
                update_seed(n * 1234)
                # scores = model(return_loss=False, **data)
                video = sample['frames']
                target = sample['label'].cuda()
                video = video.cuda(args.gpu, non_blocking=True)
                
                # tackle video
                if is_video: # test_dense
                    batch_size, clips_per_sample = video.shape[0], video.shape[1]
                    video = video.flatten(0, 1).contiguous()
                    
                if amp is not None:
                    with torch.cuda.amp.autocast():
                        scores = model.module.forward_test(video, target, **fwd_kwargs)
                else:
                    raise NotImplementedError()
                    
                if is_video:
                    scores = softmax(scores).view(batch_size, clips_per_sample, -1).mean(1) # mean of all clips
                
                
                
                # gather results
                all_scores.append(np.expand_dims(scores, axis=-1))
                
        all_scores = np.concatenate(all_scores, axis=-1)  # (B, C, T)
        # compute the uncertainty
        uncertainty = compute_uncertainty(all_scores, method=uncertainty)
        all_uncertainties.append(uncertainty)

        # compute the predictions and save labels
        mean_scores = np.mean(all_scores, axis=-1)
        preds = np.argmax(mean_scores, axis=1)
        all_results.append(preds)
        conf = np.max(mean_scores, axis=1)
        all_confidences.append(conf)

        labels = sample['label'].numpy()
        all_gts.append(labels)

        # use the first key as main key to calculate the batch size
        # batch_size = len(next(iter(data.values())))
        # for _ in range(batch_size):
        #     prog_bar.update()
    
    all_confidences = np.concatenate(all_confidences, axis=0)    
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_results = np.concatenate(all_results, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    return all_confidences, all_uncertainties, all_results, all_gts

def compute_uncertainty(predictions, method='BALD'):
    """Compute the entropy
       scores: (B x C x T)
    """
    expected_p = np.mean(predictions, axis=-1)  # mean of all forward passes (C,)
    entropy_expected_p = - np.sum(xlogy(expected_p, expected_p), axis=1)  # the entropy of expect_p (across classes)
    if method == 'Entropy':
        uncertain_score = entropy_expected_p
    elif method == 'BALD':
        expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1), axis=-1)  # mean of entropies (across classes), (scalar)
        uncertain_score = entropy_expected_p - expected_entropy
    else:
        raise NotImplementedError
    if not np.all(np.isfinite(uncertain_score)):
        uncertain_score[~np.isfinite] = 9999
    return uncertain_score


def calculate_openness(results, thresh, 
                       num_rand, # the number of random selection for ood classes
                       ind_ncls, # the number of classes in known dataset
                       ood_ncls, # the number of classes in unknwon dataset
                       logger,
                       ):
    
    # results = np.load(result_file, allow_pickle=True)
    ind_uncertainties = results['ind_unctt']  # (N1,)
    ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']
    
    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)
    # open-set auc-roc (binary class)
    preds = np.concatenate((ind_results, ood_results), axis=0)
    uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
    preds[uncertains > thresh] = 1
    preds[uncertains <= thresh] = 0
    labels = np.concatenate((np.zeros_like(ind_labels), np.ones_like(ood_labels)))
    aupr = roc_auc_score(labels, preds)
    logger.add_line(f'Threshold value: {thresh} ClosedSet Accuracy (multi-class): {acc*100}, OpenSet AUC (bin-class): {aupr * 100}')
    close_set_acc =  acc * 100
    open_set_auc = aupr * 100
    
    # open set F1 score (multi-class)
    ind_results[ind_uncertainties > thresh] = ind_ncls  # falsely rejection
    macro_F1_list = [f1_score(ind_labels, ind_results, average='macro')]
    std_list = [0]
    openness_list = [0]
    for n in range(ood_ncls):
        ncls_novel = n + 1
        openness = (1 - np.sqrt((2 * ind_ncls) / (2 * ind_ncls + ncls_novel))) * 100
        openness_list.append(openness)
        # randoml select the subset of ood samples
        macro_F1_multi = np.zeros((num_rand), dtype=np.float32)
        for m in range(num_rand):
            cls_select = np.random.choice(ood_ncls, ncls_novel, replace=False) 
            ood_sub_results = np.concatenate([ood_results[ood_labels == clsid] for clsid in cls_select])
            ood_sub_uncertainties = np.concatenate([ood_uncertainties[ood_labels == clsid] for clsid in cls_select])
            ood_sub_results[ood_sub_uncertainties > thresh] = ind_ncls  # correctly rejection
            ood_sub_labels = np.ones_like(ood_sub_results) * ind_ncls
            # construct preds and labels
            preds = np.concatenate((ind_results, ood_sub_results), axis=0)
            labels = np.concatenate((ind_labels, ood_sub_labels), axis=0)
            macro_F1_multi[m] = f1_score(labels, preds, average='macro')
        macro_F1 = np.mean(macro_F1_multi)
        std = np.std(macro_F1_multi)
        macro_F1_list.append(macro_F1)
        std_list.append(std)

    # draw comparison curves
    macro_F1_list = np.array(macro_F1_list)
    std_list = np.array(std_list)
    # plt.plot(openness_list, macro_F1_list * 100, style, linewidth=2)
    # plt.fill_between(openness_list, macro_F1_list - std_list, macro_F1_list + std_list, style)

    w_openness = np.array(openness_list) / 100.
    open_maF1_mean = np.sum(w_openness * macro_F1_list) / np.sum(w_openness)
    open_maF1_std = np.sum(w_openness * std_list) / np.sum(w_openness)
    logger.add_line(f'Threshold value: {thresh} Open macro-F1 score: {open_maF1_mean * 100}, std={open_maF1_std * 100}')
    
    return openness_list, macro_F1_list, std_list, close_set_acc, open_set_auc
