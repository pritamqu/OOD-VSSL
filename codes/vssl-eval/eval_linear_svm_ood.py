import os
import time
import torch
import warnings
import torch.multiprocessing as mp
import yaml
from tools import environment as environ
import argparse
from tools.utils import sanity_check, set_deterministic
from tools import paths
from tools import Feature_Bank, set_grad
from datasets.augmentations import get_vid_aug
from datasets import get_dataset, dataloader, FetchSubset        
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics as sk_metrics
from sklearn import preprocessing
import pickle
from models import VideoViT, SVMWrapper
from collections import OrderedDict
from tools import paths

def get_args():
        
    parser = argparse.ArgumentParser()
    
    # some stuff
    parser.add_argument("--parent_dir", default="VSSL", help="output folder name",)    
    parser.add_argument("--sub_dir", default="svm", help="output folder name",)    
    parser.add_argument("--job_id", type=str, default='00', help="jobid=%j")
    parser.add_argument("--server", type=str, default="local", help="location of server",)

    ## debug mode
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=2)
    
    ## dir stuff
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument("--output_dir", default="", help="path where to save")
    parser.add_argument("--resume", default="", help="path where to resume")
    parser.add_argument('--log_dir', type=str, default="")
    parser.add_argument('--ckpt_dir', type=str, default="")

    ## dist training stuff
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use., default to 0 while using 1 gpu')    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dist-url', default="env://", type=str, help='url used to set up distributed training, change to; "tcp://localhost:15475"')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    
    parser.add_argument("--checkpoint_path", default="", help="checkpoint_path for system restoration")

    ## pretrained model
    parser.add_argument("--weight_path", default="/mnt/PS6T/github/Anonymous-OOD-VSSL/weights/VideoBYOL_kinetics400.pth.tar", help="checkpoint_path for backbone restoration.")

    ## dataset and config
    parser.add_argument("--db", default="ucf101", help="target db", choices=['ucf101', 'hmdb51', 'kinetics400', 'mitv2', ])  
    parser.add_argument('-c', '--config-file', type=str, help="pass config file name", default="supervised_ucf_hmdb.yaml")
    
    ## sanity
    args = parser.parse_args()
    args = sanity_check(args)
    set_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True 
    
    return args


def main(args):
    
    cfg = yaml.safe_load(open(args.config_file))
    
    print(args)
    print(cfg)
        
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print(f'number of gpus per node {ngpus_per_node} - Rank {args.rank}')
    
    if args.multiprocessing_distributed:
        print('mp.spawn calling main_worker')
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        print('direct calling main_worker')
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):
        
    # Setup environment
    args.gpu = gpu
    args = environ.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, wandb_writter = environ.prep_environment_ddp(args, cfg) # prep_environment
    path = args.weight_path
    if os.path.isfile(path):
        if args.gpu is None:
            state = torch.load(path)
        else:
            state = torch.load(path, map_location='cuda:{}'.format(args.gpu))

    else:
        raise FileNotFoundError (f'weight is not found at {path}')

    linear_svm(args, cfg, state, logger=logger, tb_writter=tb_writter, wandb_writter=wandb_writter)


def linear_svm(args, cfg, backbone_state_dict, logger, tb_writter, wandb_writter):
    
    global SEED
    SEED=args.seed
    
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
        
    # get model
    model=SVMWrapper(backbone=VideoViT(**cfg['model']['backbone']), 
                     feat_op=cfg['model']['fwd_kwargs']['feat_op'], use_amp=True)
    
    state = OrderedDict()
    for key in backbone_state_dict:
        val = backbone_state_dict[key]
        if key.startswith('encoder.'):
            state[key.replace('encoder.', '')] = val
        elif key.startswith('encoder_'):
            state[key.replace('encoder_', '')] = val
        elif key == 'enc_pos_embed':
            state['pos_embed'] = val
        elif key.startswith('cuboid_embed'):
            state[key.replace('cuboid_embed', 'patch_embed')] = val
        else:
            state[key] = val
            
    ## load weights
    # model.backbone.load_state_dict(state, strict=True)
    msg = model.backbone.load_state_dict(state, strict=False)
    print(msg)
    # set grad false
    set_grad(model, requires_grad=False)
    model.eval() # when extracting features it's important to set in eval mode
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # model = torch.nn.DataParallel(model)
        
    logger.add_line(str(model)) 
    # transformations
    train_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['train']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'],
                                    )

    val_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['test']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['test']['vid_aug_kwargs'])  


    # dataset; sanity check
    if cfg['dataset']['name'] == 'kinetics_subset':
        assert cfg['dataset']['test']['subset'] == cfg['dataset']['train']['subset']
        if cfg['dataset']['train']['subset'] in ['kinetics400', 'mimetics50', 'mimetics10', 'drone']:
            args.data_dir = paths.my_paths(args.server, 'kinetics400')[-1]
        elif cfg['dataset']['train']['subset'] in ['kinetics700', 'actor_shift']:
            args.data_dir = paths.my_paths(args.server, 'kinetics700')[-1]
        else:
            raise ValueError
    elif cfg['dataset']['name'] == 'mitv2':
        assert cfg['dataset']['test']['subset'] == cfg['dataset']['train']['subset']
    elif cfg['dataset']['name'] in ['hmdb-ucf', 'ucf-hmdb']:
        pass
    else:
        raise NotImplementedError(f"{cfg['dataset']['name']} not implemented")
        
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=train_transformations, 
                                split='train')

    
        
    val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=val_transformations, 
                                split='test')
    
    ########### setup ood dataset
    ood_db = cfg['dataset']['ood_test']['name']
    ind_db = cfg['dataset']['name']
    cfg['dataset']['name'] = ood_db
    ood_val_dataset = get_dataset(root=paths.my_paths(args.server, ood_db)[-1],
                                dataset_kwargs=cfg['dataset'],
                                video_transform=get_vid_aug(
                                    name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['ood_test']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['ood_test']['vid_aug_kwargs']),
                                split='ood_test')
    
    logger.add_line("train_dataset")  
    logger.add_line(train_dataset.__repr__())    
    logger.add_line("val_dataset")  
    logger.add_line(val_dataset.__repr__())   
    logger.add_line("ood_val_dataset")  
    logger.add_line(ood_val_dataset.__repr__())   
    
    
    # back to old name
    cfg['dataset']['name'] = ind_db
                
    # adjusting batch size as test is done in dense mode
    test_batch_size = max(cfg['dataset']['batch_size'] // (3*cfg['dataset']['test']['clips_per_video']), 1)

    if args.debug:
        train_dataset = FetchSubset(train_dataset, 64)
        val_dataset = FetchSubset(val_dataset, 8)
        ood_val_dataset = FetchSubset(ood_val_dataset, 8)
        test_batch_size = 2
        cfg['dataset']['batch_size'] = 4
        
    logger.add_line(f'test batch size is {test_batch_size*args.world_size}')
    args.effective_batch = cfg['dataset']['batch_size']*args.world_size
    logger.add_line(f"train batch size is {args.effective_batch}")
    logger.add_line(f'Training dataset size: {len(train_dataset)} - Validation dataset size: {len(val_dataset)}')
            
    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset, 
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)
    
    test_loader = dataloader.make_dataloader(dataset=val_dataset, 
                                          batch_size=test_batch_size,
                                          use_shuffle=cfg['dataset']['test']['use_shuffle'],
                                          drop_last=cfg['dataset']['test']['drop_last'],
                                          num_workers=cfg['num_workers'],
                                          distributed=args.distributed)
    
    ood_test_loader = dataloader.make_dataloader(dataset=ood_val_dataset, 
                                          batch_size=test_batch_size,
                                          use_shuffle=cfg['dataset']['ood_test']['use_shuffle'],
                                          drop_last=cfg['dataset']['ood_test']['drop_last'],
                                          num_workers=cfg['num_workers'],
                                          distributed=args.distributed)

    feat_bank = Feature_Bank(args.world_size, args.distributed, model, logger, mode='vid', l2_norm=False) # setting l2-norm false, as we added l2-norm in Vid_Wrapper.

    if True:
        logger.add_line("computing features")
        end = time.time()
        train_features, train_labels, train_indexs = feat_bank.fill_memory_bank(train_loader)
        logger.add_line(f'time spent for train feat extraction: {time.time() - end}')
        end = time.time()
        val_features, val_labels, val_indexs = feat_bank.fill_memory_bank(test_loader)
        logger.add_line(f'time spent for val feat extraction: {time.time() - end}')
        end = time.time()
        ood_val_features, ood_val_labels, ood_val_indexs = feat_bank.fill_memory_bank(ood_test_loader)
        logger.add_line(f'time spent for ood_val feat extraction: {time.time() - end}')

        train_features, train_labels, train_indexs = train_features.numpy(), train_labels.numpy(), train_indexs.numpy()
        val_features, val_labels, val_indexs = val_features.numpy(), val_labels.numpy(), val_indexs.numpy()
        ood_val_features, ood_val_labels, ood_val_indexs = ood_val_features.numpy(), ood_val_labels.numpy(), ood_val_indexs.numpy()

    best_results={}
    ood_best_results={}
    logger.add_line("Running SVM...")
    logger.add_line(f"train_feat size: {train_features.shape}")     
    logger.add_line(f"val_feat size: {val_features.shape}")
    logger.add_line(f"ood_val_feat size: {ood_val_features.shape}")
    
    if isinstance(cfg['model']['svm']['cost'], list):
        pass
    else:
        cfg['model']['svm']['cost'] = [cfg['model']['svm']['cost']]
        
    for cost in cfg['model']['svm']['cost']:
        results = _compute(cost, cfg, logger,
         train_features, train_labels, train_indexs, 
         val_features, val_labels, val_indexs,
         ood_val_features, ood_val_labels, ood_val_indexs,
         )
        
        if tb_writter is not None:
            for result in results:
                tb_writter.add_scalar(f'Epoch/{result}', results[result], cost)

        if wandb_writter is not None:
            for result in results:
                wandb_writter.log({f'Epoch/{result}': results[result], 'custom_step': cost})
                
        # InD
        if len(best_results) == 0:
            best_results = results
        if 'acc' in cfg['metrics']:
            if results['test_top1'] > best_results['test_top1']:
                best_results = results
            elif results['test_top5'] > best_results['test_top5'] and  results['test_top1'] == best_results['test_top1']:
                best_results = results
        elif 'mAP' in cfg['metrics']:
            raise NotImplementedError()
            # if results['test_mean_ap'] > best_results['test_mean_ap']:
            #     best_results = results
        
        # OoD
        if len(ood_best_results) == 0:
            ood_best_results = results
        if 'acc' in cfg['metrics']:
            if results['ood_test_top1'] > ood_best_results['ood_test_top1']:
                ood_best_results = results
            elif results['ood_test_top5'] > ood_best_results['ood_test_top5'] and  results['ood_test_top1'] == ood_best_results['ood_test_top1']:
                ood_best_results = results
        elif 'mAP' in cfg['metrics']:
            raise NotImplementedError()
            # if results['test_mean_ap'] > ood_best_results['test_mean_ap']:
            #     ood_best_results = results
                
        
    for best_result in best_results:
        logger.add_line(f'Best {best_result}: {best_results[best_result]}')
        if tb_writter is not None:
            tb_writter.add_scalar(f'Epoch-Best/{best_result}', best_results[best_result], 0)
        if wandb_writter is not None:
            wandb_writter.log({f'Epoch-Best/{best_result}': best_results[best_result], 'custom_step': 0})
            
    for best_result in ood_best_results:
        logger.add_line(f'OOD-Best {best_result}: {ood_best_results[best_result]}')
        if tb_writter is not None:
            tb_writter.add_scalar(f'Epoch-OOD-Best/{best_result}', ood_best_results[best_result], 0)
        if wandb_writter is not None:
            wandb_writter.log({f'Epoch-OOD-Best/{best_result}': ood_best_results[best_result], 'custom_step': 0})
            
            
    torch.cuda.empty_cache()       
    return

def _compute(cost, cfg, logger,
             train_features, train_labels, train_indexs, 
             val_features, val_labels, val_indexs, 
             ood_val_features, ood_val_labels, ood_val_indexs,
             test_phase='test_dense'):
    
    # normalize
    if cfg['model']['svm']['scale_features']:   
        scaler = preprocessing.StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)   
    
    classifier = LinearSVC(C=cost, max_iter=cfg['model']['svm']['iter'], random_state=SEED)
    classifier.fit(train_features, train_labels.ravel())
    pred_train = classifier.decision_function(train_features)
    # for test dense, assuming this is default test case
    # reshape the data video --> cips
    if test_phase=='test_dense':
        total_samples, clips_per_sample = val_features.shape[0], val_features.shape[1]
        val_features = val_features.reshape(total_samples*clips_per_sample, -1)
        
        ood_total_samples, ood_clips_per_sample = ood_val_features.shape[0], ood_val_features.shape[1]
        ood_val_features = ood_val_features.reshape(ood_total_samples*ood_clips_per_sample, -1)
        
    # scale if true
    if cfg['model']['svm']['scale_features']:
        val_features = scaler.transform(val_features)
        ood_val_features = scaler.transform(ood_val_features)
    # predict
    pred_test = classifier.decision_function(val_features)
    ood_pred_test = classifier.decision_function(ood_val_features)
    
    if test_phase=='test_dense':
        pred_test = pred_test.reshape(total_samples, clips_per_sample, -1).mean(1)
        ood_pred_test = ood_pred_test.reshape(ood_total_samples, ood_clips_per_sample, -1).mean(1)

    if 'acc' in cfg['metrics']:
        metrics = compute_accuracy_metrics(pred_train, train_labels[:, None], prefix='train_')
        metrics.update(compute_accuracy_metrics(pred_test, val_labels[:, None], prefix='test_'))
        metrics.update(compute_accuracy_metrics(ood_pred_test, ood_val_labels[:, None], prefix='ood_test_'))
    if 'mAP' in cfg['metrics']:
        raise NotImplementedError()
        
    logger.add_line(f"Video Linear SVM on {cfg['dataset']['name']} cost: {cost}")
    for metric in metrics:
        logger.add_line(f"{metric}: {metrics[metric]}") 
        
    # return metrics['test_top1'], metrics['test_top5']
    return metrics
             

def compute_accuracy_metrics(pred, gt, prefix=''):
  order_pred = np.argsort(pred, axis=1)
  assert len(gt.shape) == len(order_pred.shape) == 2
  top1_pred = order_pred[:, -1:]
  top5_pred = order_pred[:, -5:]
  top1_acc = np.mean(top1_pred == gt)
  top5_acc = np.mean(np.max(top5_pred == gt, 1))
  return {prefix + 'top1': top1_acc*100,
          prefix + 'top5': top5_acc*100}


    
if __name__ == "__main__":
    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)

