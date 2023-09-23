import os
import time
import torch
import warnings
import torch.multiprocessing as mp
import yaml
from tools import environment as environ
import argparse
from tools.utils import sanity_check, set_deterministic
import numpy as np
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math
import sys
from tools import AverageMeter, ProgressMeter, accuracy, warmup_cosine_scheduler, warmup_multistep_scheduler, get_params_groups, Classifier
from datasets.augmentations import get_vid_aug
from datasets import get_dataset, dataloader, FetchSubset        
from checkpointing import create_or_restore_training_state3, commit_state3
from models import has_batchnorms
from models import VideoViT
from collections import OrderedDict
from tools import paths
GB = (1024*1024*1024)


def get_args():
        
    parser = argparse.ArgumentParser()
    
    # some stuff
    parser.add_argument("--parent_dir", default="VSSL", help="output folder name",)    
    parser.add_argument("--sub_dir", default="finetune", help="output folder name",)    
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
    parser.add_argument("--db", default="ucf101", help="target db", choices=['ucf101', 'hmdb51', 'kinetics400', 'mitv2'])  
    parser.add_argument('-c', '--config-file', type=str, help="pass config file name", 
                        default="byol_ucf_hmdb.yaml",
                        )

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
        cfg['dataset']['batch_size'] = 2

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

    finetune(args, cfg, state, ngpus_per_node, logger=logger, tb_writter=tb_writter, wandb_writter=wandb_writter)


def finetune(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter):
    
    if args.debug:
        cfg['eval_freq'] = 1
        cfg, args = environ.set_debug_mode(cfg, args)
        
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
    msg = model.load_state_dict(state, strict=False)
    print(msg)

    model.head = Classifier(feat_dim=model.embed_dim, **cfg['model']['classifier'])
    
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


    # dataset
    if cfg['dataset']['name'] == 'kinetics_subset':
        assert cfg['dataset']['test']['subset'] == cfg['dataset']['train']['subset']
        if cfg['dataset']['train']['subset'] in ['kinetics400', 'mimetics50', 'mimetics10']:
            args.data_dir = paths.my_paths(args.server, 'kinetics400')[-1]
        elif cfg['dataset']['train']['subset'] in ['kinetics700', 'actor_shift']:
            args.data_dir = paths.my_paths(args.server, 'kinetics700')[-1]
        else:
            raise ValueError
    elif cfg['dataset']['name'] == 'mitv2':
        assert cfg['dataset']['test']['subset'] == cfg['dataset']['train']['subset']
    elif cfg['dataset']['name'] in ['ucf-hmdb', 'hmdb-ucf']:
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
        train_dataset = FetchSubset(train_dataset, 16)
        val_dataset = FetchSubset(val_dataset, 4)
        ood_val_dataset = FetchSubset(ood_val_dataset, 4)
        test_batch_size = 8
        
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
    
    # mixup
    mixup_fn = None
    mixup_active = cfg['hyperparams']['mixup'] > 0 or cfg['hyperparams']['cutmix'] > 0. or cfg['hyperparams']['cutmix_minmax'] is not None
    if mixup_active:
        logger.add_line("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=cfg['hyperparams']['mixup'], 
            cutmix_alpha=cfg['hyperparams']['cutmix'], 
            cutmix_minmax=cfg['hyperparams']['cutmix_minmax'],
            prob=cfg['hyperparams']['mixup_prob'], switch_prob=cfg['hyperparams']['mixup_switch_prob'], mode=cfg['hyperparams']['mixup_mode'],
            label_smoothing=cfg['hyperparams']['label_smoothing'], num_classes=cfg['model']['classifier']['num_classes'])
    else:
        logger.add_line("Mixup is deactivated!")
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif cfg['hyperparams']['label_smoothing'] > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg['hyperparams']['label_smoothing'])
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    test_criterion = torch.nn.CrossEntropyLoss()
        
        
    # optim
    param_groups = get_params_groups(model.module, 
                                     weight_decay=cfg['hyperparams']['weight_decay'], 
                                     no_weight_decay_list=model.module.no_weight_decay(), 
                                     layer_decay=cfg['hyperparams']['layer_decay'])
    
    optimizer = torch.optim.AdamW(param_groups, 
                                  betas=cfg['hyperparams']['optimizer']['betas'],
                                  lr=cfg['hyperparams']['lr']['base_lr']) # lr will over ride by lr_scheduler
    # lr scheduler
    if cfg['hyperparams']['lr']['name'] == 'cosine':
        lr_scheduler = warmup_cosine_scheduler(base_lr=cfg['hyperparams']['lr']['base_lr'], 
                                        final_lr=cfg['hyperparams']['lr']['final_lr'], 
                                        num_epochs=cfg['hyperparams']['num_epochs'], 
                                        iter_per_epoch=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        warmup_lr=cfg['hyperparams']['lr']['warmup_lr'])
    elif cfg['hyperparams']['lr']['name'] == 'fixed':
        iters = cfg['hyperparams']['num_epochs'] * len(train_loader)
        lr_scheduler = np.ones(iters) * cfg['hyperparams']['lr']['base_lr']
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
    if args.debug:
        end_epoch=3
    
   
    best_top1, best_top5=0, 0
    best_epoch=start_epoch
    tr_top1, tr_top5 = 0, 0
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        fwd_kwargs = cfg['model']['fwd_kwargs']
        
        ## train
        marker = time.time()
        tr_top1, tr_top5 = run_phase(phase='train', 
                                      loader=train_loader, 
                                      model=model,
                                      optimizer=optimizer, 
                                      criterion=criterion, 
                                      mixup_fn=mixup_fn,
                                      fwd_kwargs=fwd_kwargs,
                                      lr_scheduler=lr_scheduler, 
                                      wd_scheduler=None, 
                                      amp=amp,
                                      epoch=epoch, 
                                      args=args, logger=logger, 
                                      tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                      print_freq=cfg['progress']['print_freq'])     
        
        logger.add_line(f'trainng took {time.time()-marker} seconds')
        marker = time.time()

        if args.rank==0:
            logger.add_line('saving model')    
            commit_state3(args, model, optimizer, epoch, amp, rng, logger)
                
        if args.rank==0 and ((epoch+1==end_epoch) or (epoch+1)%cfg['eval_freq']==0):
            model_path = os.path.join(args.ckpt_dir, "model.pth.tar")
            torch.save(model.module.state_dict(), model_path)
            print(f"model backbone is saved to \n{args.ckpt_dir}")       

        logger.add_line(f'model saving took {time.time()-marker} seconds')
        marker = time.time()
        
        if tb_writter is not None:
            tb_writter.add_scalar('fine_tune_epoch/tr_top1', tr_top1, epoch)
            tb_writter.add_scalar('fine_tune_epoch/tr_top5', tr_top5, epoch)
        if wandb_writter is not None:
            wandb_writter.log({'fine_tune_epoch/tr_top1': tr_top1, 'custom_step': epoch})
            wandb_writter.log({'fine_tune_epoch/tr_top5': tr_top5, 'custom_step': epoch})
            
        ## test
        if (epoch+1) % cfg['eval_freq'] == 0 or (epoch+1) == end_epoch:
            vid_top1, vid_top5 = run_phase(phase='test_dense', 
                                          loader=test_loader, 
                                          model=model,
                                          optimizer=None, 
                                          criterion=test_criterion, # criterion
                                          mixup_fn=None,
                                          fwd_kwargs=fwd_kwargs,
                                          lr_scheduler=None, 
                                          wd_scheduler=None, 
                                          amp=amp,
                                          epoch=epoch, 
                                          args=args, logger=logger, 
                                          tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                          print_freq=cfg['progress']['print_freq'])

            logger.add_line(f'validation took {time.time()-marker} seconds')
            top1, top5 = vid_top1, vid_top5
            if top1>=best_top1:
                best_top1=top1
                best_top5=top5
                best_epoch=epoch
                # Save checkpoint
                best_ind_model=os.path.join(args.ckpt_dir, f"best_model_{ind_db}.pth.tar")
                if args.rank==0:
                    torch.save(model.module.state_dict(), best_ind_model)
                
            if tb_writter is not None:
                tb_writter.add_scalar('fine_tune_epoch/vid_top1', vid_top1, epoch)
                tb_writter.add_scalar('fine_tune_epoch/vid_top5', vid_top5, epoch)
            if wandb_writter is not None:
                wandb_writter.log({'fine_tune_epoch/vid_top1': vid_top1, 'custom_step': epoch})
                wandb_writter.log({'fine_tune_epoch/vid_top5': vid_top5, 'custom_step': epoch})
                
                
    # ood test at the end
    # load the best ind model
    best_state = torch.load(best_ind_model, map_location='cuda:{}'.format(args.gpu))
    model.module.load_state_dict(best_state)
    ood_vid_top1, ood_vid_top5 = run_phase(phase='test_dense', 
                                    loader=ood_test_loader, 
                                    model=model,
                                    optimizer=None, 
                                    criterion=test_criterion, # criterion
                                    mixup_fn=None,
                                    fwd_kwargs=fwd_kwargs,
                                    lr_scheduler=None, 
                                    wd_scheduler=None, 
                                    amp=amp,
                                    epoch=epoch, 
                                    args=args, logger=logger, 
                                    tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                    print_freq=cfg['progress']['print_freq'])

    logger.add_line(f'ood validation took {time.time()-marker} seconds')
        
    if tb_writter is not None:
        tb_writter.add_scalar('fine_tune_epoch/ood_vid_top1', ood_vid_top1, epoch)
        tb_writter.add_scalar('fine_tune_epoch/ood_vid_top5', ood_vid_top5, epoch)
    if wandb_writter is not None:
        wandb_writter.log({'fine_tune_epoch/ood_vid_top1': ood_vid_top1, 'custom_step': epoch})
        wandb_writter.log({'fine_tune_epoch/ood_vid_top5': ood_vid_top5, 'custom_step': epoch})                
                                  
    # --------- end log               
    if args.rank==0:
        logger.add_line(f'InD Acc top1: {best_top1} - top5: {best_top5}')
        logger.add_line(f'OOD Acc top1: {ood_vid_top1} - top5: {ood_vid_top5}')
                    
    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()
        
    return



def run_phase(phase, loader, model, optimizer, criterion, mixup_fn, fwd_kwargs,
              lr_scheduler, wd_scheduler, amp,
              epoch, args, logger, tb_writter, wandb_writter, print_freq):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    max_lr_meter = AverageMeter('Max_LR', ':.4e', 0)
    min_lr_meter = AverageMeter('Min_LR', ':.4e', 0)
    weight_decay_meter = AverageMeter('WD', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    if phase=='train':
        progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, 
                                                      max_lr_meter, min_lr_meter, weight_decay_meter,
                                                      top1_meters, top5_meters, gpu_meter,], phase=phase, epoch=epoch, logger=logger)
    else:
        progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, 
                                                      top1_meters, top5_meters,], phase=phase, epoch=epoch, logger=logger)

    

    if phase == 'train':
        model.train()
    else:
        model.eval()

    end = time.time()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        
        step = epoch * len(loader) + it
        # update lr during training
        if phase =='train':
            
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr_scheduler[step] * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr_scheduler[step]
                if wd_scheduler is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_scheduler[step]

        # prepare data
        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()

        elif phase == 'train':
            if mixup_fn is not None:
                video, target = mixup_fn(video, target)


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
            if criterion is not None:
                loss = criterion(logits, target_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)
            
        if phase == 'train':
            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training") # for log
                logger.add_line(f"Loss is {loss.item()}, stopping training") # for logger
                sys.exit(1)
            
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
        
        # with torch.no_grad():
        if phase == 'train':
            loss_meters.update(loss.item(), target.size(0))
            if mixup_fn is not None:
                # if mixup is enabled, skipping acc calculation during training
                acc1, acc5 = torch.tensor([0.]).cuda(args.gpu), torch.tensor([0.]).cuda(args.gpu)
            else:
                acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
        elif phase == 'test_dense':
            if criterion is not None:
                loss_meters.update(loss.item(), target.size(0))
            acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
        
        top1_meters.update(acc1[0].item(), target.size(0))
        top5_meters.update(acc5[0].item(), target.size(0))

        if phase == 'train':
            # lr meter
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
    
            max_lr_meter.update(max_lr, target.size(0))
            min_lr_meter.update(min_lr, target.size(0))
            
            # wd meter
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
                    weight_decay_meter.update(weight_decay_value, target.size(0))
        
        # log
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)
            if tb_writter is not None :
                if phase == 'train':
                    for meter in progress.meters:
                        tb_writter.add_scalar(f'{phase}_fine_tune_iter/{meter.name}', meter.val, step)
            
            if wandb_writter is not None : 
                if phase == 'train':
                    for meter in progress.meters:
                         wandb_writter.log({f'{phase}_fine_tune_iter/{meter.name}': meter.val, 'custom_step': step})
            
    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    return top1_meters.avg, top5_meters.avg


if __name__ == "__main__":
    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)
