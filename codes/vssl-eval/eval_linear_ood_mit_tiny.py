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
import os
import torch
import time
import numpy as np
from tools import AverageMeter, ProgressMeter, accuracy, warmup_cosine_scheduler, warmup_multistep_scheduler, warmup_fixed_scheduler, Classifier
from datasets.augmentations import get_vid_aug
from tools import environment as environ
from checkpointing import commit_state3, create_or_restore_training_state3
from models import VideoViT
from models import has_batchnorms
from collections import OrderedDict
import copy
from einops import rearrange
import torch.nn as nn
from tools import paths
GB = (1024*1024*1024)
from datasets import dataloader, FetchSubset, get_dataset    
# from datasets.feat_loaders import FeatDataset, FeatDataset_ML
from tools import paths, mean_ap_metric, calculate_prec_recall_f1
from tools import synchronize_holder

def get_args():
        
    parser = argparse.ArgumentParser()
    
    # some stuff
    parser.add_argument("--parent_dir", default="VSSL", help="output folder name",)    
    parser.add_argument("--sub_dir", default="linear", help="output folder name",)    
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
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))

    ## dist training stuff
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use., default to 0 while using 1 gpu')    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dist-url', default="env://", type=str, help='url used to set up distributed training, change to; "tcp://localhost:15475"')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    
    parser.add_argument("--checkpoint_path", default="", help="checkpoint_path for system restoration")
    parser.add_argument('--checkpoint_interval', default=3600, type=int, help='checkpoint_interval')
    
    ## pretrained model
    parser.add_argument("--weight_path", default="/mnt/PS6T/github/Anonymous-OOD-VSSL/weights/VideoBYOL_kinetics400.pth.tar", help="checkpoint_path for backbone restoration.")

    ## dataset and config
    parser.add_argument("--db", default="mitv2", help="target db", 
                        choices=['kinetics400', 'mitv2'])  
    parser.add_argument('-c', '--config-file', type=str, help="pass config file name", 
                        default="byol_mitv2_tiny.yaml")

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
        # cfg['dataset']['batch_size'] = 2

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
            # Map model to be loaded to specified single gpu.
            state = torch.load(path, map_location='cuda:{}'.format(args.gpu))
    else:
        raise FileNotFoundError (f'weight is not found at {path}')

    linear(args, cfg, state, ngpus_per_node, logger=logger, tb_writter=tb_writter, wandb_writter=wandb_writter)


def linear(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter):
    
    if args.debug:
        pass
                
    # get the model
    model=VideoViT(**cfg['model']['backbone'])    
    # fix the names to the more standard one.
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
               
    # load weights
    # model.load_state_dict(state, strict=True)
    msg = model.load_state_dict(state, strict=False)
    print(msg)
    model.head = Classifier(feat_dim=model.embed_dim, **cfg['model']['classifier'])

    
    # freeze all but the classifier head for linear eval
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
     
    # check no. of model params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.add_line('number of params (M): %.2f' % (n_parameters / 1.e6))
    # sync bn if used
    if args.distributed and cfg['sync_bn']:
        if has_batchnorms(model):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # distribute the model
    model = model.cuda(args.gpu)
    model, _, cfg['dataset']['batch_size'], cfg['num_workers'] = environ.distribute_model_to_cuda(models=model, 
                                                                     args=args, 
                                                                     batch_size=cfg['dataset']['batch_size'], 
                                                                     num_workers=cfg['num_workers'], 
                                                                     ngpus_per_node=ngpus_per_node)
    
    
    # dataset
    # feat_dir = None
    if cfg['dataset']['name'] == 'mitv2':
        assert cfg['dataset']['test']['subset'] == cfg['dataset']['train']['subset']
        assert cfg['dataset']['ood_test']['name']=='tinyvirat'
        # feat_dir = os.path.join(os.path.dirname(os.path.dirname(args.weight_path)), f"feats_{cfg['dataset']['name']}")
        # if os.path.exists(feat_dir):
        #     args.data_dir = feat_dir
    else:
        raise NotImplementedError(f"{cfg['dataset']['name']} not implemented")

    # transformations
    # if feat_dir is None:
    # logger.add_line(f"Loaidng features from {feat_dir} for dataset {cfg['dataset']['name']}")
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



        
    # if feat_dir is not None:
    #     train_dataset = FeatDataset(root=args.data_dir, split='train')
    #     val_dataset = FeatDataset(root=args.data_dir, split='val')
    # else:
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
    
    # feat_dir = None
    # if ood_db == 'tinyvirat':
    #     feat_dir = os.path.join(os.path.dirname(os.path.dirname(args.weight_path)), f"feats_{ood_db}")
    #     if os.path.exists(feat_dir):
    #         args.data_dir = feat_dir
    # else:
    #     raise NotImplementedError(f"{cfg['dataset']['name']} not implemented")
        
    # if feat_dir is not None:
    #     logger.add_line(f"Loaidng features from {feat_dir} for dataset {ood_db}")
    #     ood_val_dataset = FeatDataset_ML(root=args.data_dir, split='val')
    # else:
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
    
    if args.debug:
        train_dataset = FetchSubset(train_dataset, 17)
        val_dataset = FetchSubset(val_dataset, 11)
        ood_val_dataset = FetchSubset(ood_val_dataset, 11)
        test_batch_size = 2
    
    # back to old name
    cfg['dataset']['name'] = ind_db
                
    # adjusting batch size as test is done in dense mode
    test_batch_size = max(cfg['dataset']['batch_size'] // (3*cfg['dataset']['test']['clips_per_video']), 1)
        
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
    
    # optim; pass just classifier params
    if cfg['hyperparams']['optimizer']['name']=='sgd':
        optimizer = torch.optim.SGD(model.module.head.parameters(), 
                                    lr=1e-3, # setting lr through lr scheduler
                                    momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adam':
        optimizer = torch.optim.Adam(model.module.head.parameters(), 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adamw':
        optimizer = torch.optim.AdamW(model.module.head.parameters(), 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
    else:
        raise NotImplementedError()
    
    # lr scheduler
    if cfg['hyperparams']['lr']['name'] == 'cosine':
        lr_scheduler = warmup_cosine_scheduler(base_lr=cfg['hyperparams']['lr']['base_lr'], 
                                        final_lr=cfg['hyperparams']['lr']['final_lr'], 
                                        num_epochs=cfg['hyperparams']['num_epochs'], 
                                        iter_per_epoch=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        warmup_lr=cfg['hyperparams']['lr']['warmup_lr'])
    elif cfg['hyperparams']['lr']['name'] == 'fixed':
        # iters = cfg['hyperparams']['num_epochs'] * len(train_loader)
        # lr_scheduler = np.ones(iters) * cfg['hyperparams']['lr']['base_lr']
        lr_scheduler = warmup_fixed_scheduler(warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                              warmup_lr=cfg['hyperparams']['lr']['warmup_lr'], 
                                              num_epochs=cfg['hyperparams']['num_epochs'], 
                                              base_lr=cfg['hyperparams']['lr']['base_lr'], 
                                              iter_per_epoch=len(train_loader))
    elif cfg['hyperparams']['lr']['name'] == 'step':
        lr_scheduler = warmup_multistep_scheduler(base_lr=cfg['hyperparams']['lr']['base_lr'], 
                                        milestones=cfg['hyperparams']['lr']['milestones'], 
                                        gamma=cfg['hyperparams']['lr']['gamma'],                                         
                                        num_epochs=cfg['hyperparams']['num_epochs'], 
                                        iter_per_epoch=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        warmup_lr=cfg['hyperparams']['lr']['warmup_lr'])
    else:
        raise NotImplementedError()

    # use apex for mixed precision training
    if cfg['apex']:
        amp = torch.cuda.amp.GradScaler() 
    else:
        amp=None
   
    model, optimizer, start_epoch, amp, rng = create_or_restore_training_state3(args, model, optimizer, logger, amp)
        
    # Start training
    end_epoch = cfg['hyperparams']['num_epochs']
    logger.add_line('='*30 + ' Training Started Finetune' + '='*30)
    
    best_top1, best_top5=0, 0
    best_epoch=start_epoch
    tr_top1, tr_top5 = 0, 0
    
    # if feat_dir is not None:
    #     net = model.module.head
    # else:
    net = model
    
    
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        fwd_kwargs = cfg['model']['fwd_kwargs']
            
        # train
        tr_top1, tr_top5 = run_phase('train', train_loader, net, fwd_kwargs,
                                     optimizer, lr_scheduler, amp,
                                     epoch, args, logger, tb_writter, wandb_writter, cfg['progress']['print_freq'])
        
        
        if args.rank==0:
            logger.add_line('saving model')    
            commit_state3(args, model, optimizer, epoch, amp, rng, logger)
            
        # test
        acc_top1, acc_top5 = run_phase('test_dense', test_loader, net, fwd_kwargs,
                                       optimizer, lr_scheduler, amp,
                                       epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
        top1, top5 = acc_top1, acc_top5
        if top1>best_top1:
            best_top1=top1
            best_top5=top5
            best_epoch=epoch
            # Save checkpoint
            best_ind_model=os.path.join(args.ckpt_dir, f"best_model_{ind_db}.pth.tar")
            if args.rank==0:
                torch.save(model.module.state_dict(), best_ind_model)
           
        if tb_writter is not None:
            tb_writter.add_scalar('train/tr_top1', tr_top1, epoch)
            tb_writter.add_scalar('eval/te_top1', acc_top1, epoch)
            tb_writter.add_scalar('eval/te_top5', acc_top5, epoch)
                        
        if wandb_writter is not None:
            wandb_writter.log({'train/tr_top1': tr_top1, 'custom_step': epoch})
            wandb_writter.log({'eval/te_top1': acc_top1, 'custom_step': epoch})
            wandb_writter.log({'eval/te_top5': acc_top5, 'custom_step': epoch})
            
            
    # ood test at the end
    # load the best ind model
    best_state = torch.load(best_ind_model, map_location='cuda:{}'.format(args.gpu))
    model.module.load_state_dict(best_state)
    ood_acc_top1, ood_acc_top5 = run_phase_multilabel('test_dense', ood_test_loader, net, fwd_kwargs,
                                    optimizer, lr_scheduler, amp,
                                    epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'], 
                                    ensemble=cfg['model']['ensemble'])
    if tb_writter is not None:
        tb_writter.add_scalar('eval/ood_f1', ood_acc_top1, epoch)
        tb_writter.add_scalar('eval/ood_acc', ood_acc_top5, epoch)
                    
    if wandb_writter is not None:
        wandb_writter.log({'eval/ood_f1': ood_acc_top1, 'custom_step': epoch})
        wandb_writter.log({'eval/ood_acc': ood_acc_top5, 'custom_step': epoch})
        
    # --------- end log  
    logger.add_line(f'InD Acc - top1: {best_top1} - top5: {best_top5}')
    logger.add_line(f'OOD Acc - F1: {ood_acc_top1} - Acc: {ood_acc_top5}')
                      
    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()
        
    return

def run_phase(phase, loader, model, fwd_kwargs, optimizer, lr_scheduler, amp,
              epoch, args, logger, tb_writter, wandb_writter, print_freq):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, top1_meters, top5_meters, gpu_meter,], phase=phase, epoch=epoch, logger=logger)


    if phase == 'train':
        model.train()
        LOG_HEAD='train'
    else:
        model.eval()
        LOG_HEAD='eval'

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        
        # update lr during training
        if phase =='train':
            step = epoch * len(loader) + it
            for pi, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler[step]

        # prepare data
        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()
            
        # compute outputs
        if phase == 'train':
            optimizer.zero_grad()
            if amp is not None:
                with torch.cuda.amp.autocast():
                    logits = model(video, **fwd_kwargs)
            else:
                logits = model(video, **fwd_kwargs)
        else:
            with torch.no_grad():
                if amp is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(video, **fwd_kwargs)
                else:
                    logits = model(video, **fwd_kwargs)
                        
        # compute loss and measure accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, target_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)

        with torch.no_grad():
            acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
            loss_meters.update(loss.item(), target.size(0))
            top1_meters.update(acc1[0].item(), target.size(0))
            top5_meters.update(acc5[0].item(), target.size(0))

        # compute gradient
        if phase == 'train':
            if amp is not None:
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
            else:
                loss.backward()
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure gpu usage
        gpu_meter.update(torch.cuda.max_memory_allocated()/GB) 
        
        # log
        step = epoch * len(loader) + it
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)
            if tb_writter is not None:
                tb_writter.add_scalar(f'{LOG_HEAD}/LR', optimizer.param_groups[0]['lr'], step)
                for meter in progress.meters:
                    tb_writter.add_scalar(f'{LOG_HEAD}/{meter.name}', meter.val, step)
            
            if wandb_writter is not None and phase == 'train':
                wandb_writter.log({f'{LOG_HEAD}/LR': optimizer.param_groups[0]['lr'], 'custom_step': step})
                for meter in progress.meters:
                     wandb_writter.log({f'{LOG_HEAD}/{meter.name}': meter.val, 'custom_step': step})
            

    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    return top1_meters.avg, top5_meters.avg
    

def run_phase_multilabel(phase, loader, model, fwd_kwargs, optimizer, lr_scheduler, amp,
              epoch, args, logger, tb_writter, wandb_writter, print_freq, ensemble):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    # top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    # top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, gpu_meter,], phase=phase, epoch=epoch, logger=logger)


    if phase == 'train':
        model.train()
        LOG_HEAD='ml-train'
    else:
        model.eval()
        LOG_HEAD='ml-eval'

    end = time.time()
    # softmax = torch.nn.Softmax(dim=1)
    activation = torch.nn.Sigmoid()
    criterion =  torch.nn.BCEWithLogitsLoss()
    
    pred_holder = []
    target_holder = []
    results = None
    
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        
        # update lr during training
        if phase =='train':
            step = epoch * len(loader) + it
            for pi, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler[step]

        # prepare data
        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()
            
        # compute outputs
        if phase == 'train':
            optimizer.zero_grad()
            if amp is not None:
                with torch.cuda.amp.autocast():
                    logits = model(video, **fwd_kwargs)
                    # logits = model(video)
            else:
                logits = model(video, **fwd_kwargs)
                # logits = model(video)
        else:
            with torch.no_grad():
                if amp is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(video, **fwd_kwargs)
                        # logits = model(video)
                else:
                    logits = model(video, **fwd_kwargs)
                    # logits = model(video)
                        
        # compute loss and measure accuracy
        target=target.type(logits.dtype)
        if phase == 'test_dense':
            # multilabel
            confidence = activation(logits).view(batch_size, clips_per_sample, -1)
            target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample, 1).view(batch_size*clips_per_sample, -1)
            
            if ensemble=='max':
                confidence = torch.max(confidence, dim=1)[0]
            elif ensemble=='sum':
                confidence = torch.sum(confidence, dim=1)
            elif ensemble=='mean':
                confidence = torch.mean(confidence, dim=1)
            else:
                raise ValueError(f'unknown ensemble method: {ensemble}')

            if criterion is not None:
                loss = criterion(logits, target_tiled)
        else:
            confidence = activation(logits)
            loss = criterion(logits, target)

        with torch.no_grad():
            # acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
            loss_meters.update(loss.item(), target.size(0))
            # top1_meters.update(acc1[0].item(), target.size(0))
            # top5_meters.update(acc5[0].item(), target.size(0))

        # compute gradient
        if phase == 'train':
            if amp is not None:
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
            else:
                loss.backward()
                optimizer.step()
                
        # accumulate output --------- RESULT
        # if phase != 'train':
        if True:
            pred_holder.append(confidence.detach())
            if len(target.shape)==1:
                target_holder.append(target.unsqueeze(-1).detach())
            else: # for multi-label
                target_holder.append(target.detach())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure gpu usage
        gpu_meter.update(torch.cuda.max_memory_allocated()/GB) 
        
        # log
        step = epoch * len(loader) + it
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)
            if tb_writter is not None and phase == 'train':
                tb_writter.add_scalar(f'{LOG_HEAD}/LR', optimizer.param_groups[0]['lr'], step)
                for meter in progress.meters:
                    tb_writter.add_scalar(f'{LOG_HEAD}/{meter.name}', meter.val, step)
            
            if wandb_writter is not None and phase == 'train':
                wandb_writter.log({f'{LOG_HEAD}/LR': optimizer.param_groups[0]['lr'], 'custom_step': step})
                for meter in progress.meters:
                     wandb_writter.log({f'{LOG_HEAD}/{meter.name}': meter.val, 'custom_step': step})
            

    # if phase != 'train': # -------- RESULT
    if True:
        # calculating results during test
        pred_holder = torch.vstack(pred_holder)
        target_holder = torch.vstack(target_holder)
        if args.distributed:
            pred_holder = synchronize_holder(pred_holder, args.gpu)
            target_holder = synchronize_holder(target_holder, args.gpu)
        else:
            pred_holder = [pred_holder]
            target_holder = [target_holder]

        pred_holder = torch.cat(pred_holder)
        target_holder = torch.cat(target_holder) 
        # results = mean_ap_metric(pred_holder.cpu().numpy(), target_holder.cpu().numpy(), logger)
        results = calculate_prec_recall_f1(pred_holder, target_holder)

    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    
    for k in results:
        logger.add_line(f"{LOG_HEAD} EVAL: {k} - {results[k]}")
    
    return results['f1'], results['acc']

if __name__ == "__main__":
    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)
