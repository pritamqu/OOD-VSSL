import os
import time
import math
import sys
import torch
import warnings
import argparse
import yaml
import torch.multiprocessing as mp
from datasets.augmentations import get_vid_aug
from datasets import get_dataset, dataloader, FetchSubset
from tools import environment as environ
from models import get_model, has_batchnorms
from optimizers import get_optimizer_w_pred, cosine_scheduler
from tools import AverageMeter, ProgressMeter, sanity_check, set_deterministic
# from tools.utils import resume_model, save_checkpoint # general use
from checkpointing import create_or_restore_training_state2, commit_state2
import torchvision
import numpy as np
GB = (1024*1024*1024)


def get_args(mode='default'):

    parser = argparse.ArgumentParser()

    # some stuff
    parser.add_argument("--parent_dir", default="VSSL", help="output folder name",)
    parser.add_argument("--sub_dir", default="pretext", help="output folder name",)
    parser.add_argument("--job_id", type=str, default='00', help="jobid=%j")
    parser.add_argument("--server", type=str, default="local", help="location of server",)
    parser.add_argument("--db", default="kinetics400", help="target db",
                        choices=['kinetics400', 'kinetics700', 'audioset'])
    parser.add_argument('-c', '--config-file', type=str, help="config", default="moco.yaml")

    ## debug mode
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=2)

    ## dir stuff
    parser.add_argument('--data_dir', type=str, default='/mnt/PS6T/datasets')
    parser.add_argument("--output_dir", default="/mnt/PS6T/OUTPUTS", help="path where to save")
    parser.add_argument("--resume", default="", help="path where to resume")
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))

    ## dist training stuff
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use., default to 0 while using 1 gpu')
    parser.add_argument('--seed', type=int, default=774826)
    parser.add_argument('--dist-url', default="env://", type=str, help='url used to set up distributed training, change to; "tcp://localhost:15475"')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument("--checkpoint_path", default="", help="checkpoint_path for system restoration")

    args = parser.parse_args()
    args.mode = mode
    args = sanity_check(args)
    set_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True

    return args

def main(args):

    cfg = yaml.safe_load(open(args.config_file))

    # a quick test in debug mode
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
        # small model for debug
        cfg['model']['kwargs']['encoder_cfg']='tiny_encoder'
           
    print(args)
    print(cfg)
    
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
    
    assert cfg['model']['type'] == 'moco'
    # sanity
    video_frames=int(cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'])
    cfg['model']['kwargs']['num_frames'] = video_frames


    #---------------------- Setup environment
    args.gpu = gpu
    args = environ.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, wandb_writter = environ.prep_environment_ddp(args, cfg)
    # use apex for mixed precision training
    amp = torch.cuda.amp.GradScaler() if cfg['apex'] else None
    
    #---------------------- define model
    
    model = get_model(cfg['model'])
    
    # synchronize batch norm
    if args.distributed and cfg['sync_bn']:
        if has_batchnorms(model):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(args.gpu)
    # wrap in ddp
    model, args, cfg['dataset']['batch_size'], cfg['num_workers'] = environ.distribute_model_to_cuda(models=model,
                                                                         args=args,
                                                                         batch_size=cfg['dataset']['batch_size'],
                                                                         num_workers=cfg['num_workers'],
                                                                         ngpus_per_node=ngpus_per_node)
    logger.add_line(str(model.module))
    effective_batch = cfg['dataset']['batch_size'] * args.world_size
    logger.add_line(f'effective batch size: {effective_batch}')
    # model size
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.add_line('Total number of params (M): %.2f' % (n_parameters / 1.e6))
    

    vid_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=video_frames,
                                    mode=cfg['dataset']['train']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'])
    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=vid_transformations,
                                split='train')
    
    logger.add_line("train_dataset")  
    logger.add_line(train_dataset.__repr__())    

    if args.debug:
        train_dataset = FetchSubset(train_dataset, 16)

    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset,
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)
    
    # define optimizer
    predictor_prefix = ('module.predictor', 'predictor')
    def _get_params_groups(model):
        base_not_regularized = []
        base_regularized = []
        predictor_not_regularized = []
        predictor_regularized = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(predictor_prefix):
                # predictor.append(param)
                if name.endswith(".bias") or len(param.shape) == 1:
                    predictor_not_regularized.append(param)
                else:
                    predictor_regularized.append(param)
            else:
                # base.append(param)
                if name.endswith(".bias") or len(param.shape) == 1:
                    base_not_regularized.append(param)
                else:
                    base_regularized.append(param)
    
        return [
            {'name': 'base_0', 'params': base_not_regularized, 'weight_decay': 0.},
            {'name': 'base_wd', 'params': base_regularized, },
            {'name': 'predictor_0', 'params': predictor_not_regularized, 'weight_decay': 0.},
            {'name': 'predictor_wd', 'params': predictor_regularized, },
                ]

    parameters = _get_params_groups(model)
    if cfg['hyperparams']['optimizer']['name'] == 'adamw':
        optimizer = torch.optim.AdamW(parameters, betas=cfg['hyperparams']['optimizer']['betas']) # lr-scheduler will set lr
    else:
        raise NotImplementedError()
    
    
    # define EMA Scheduler
    ema_scheduler = None
    if cfg['hyperparams']['ema']['name'] == 'cosine':
        ema_scheduler = cosine_scheduler(base_value=cfg['hyperparams']['ema']['base'], 
                                        final_value=cfg['hyperparams']['ema']['final'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['ema']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['ema']['warmup'])
    else:
        raise NotImplementedError(print(f"{cfg['hyperparams']['ema']['name']} not implemented"))
    
    # define lr scheduler
    lr_scheduler = {}
    if cfg['hyperparams']['lr']['name'] == 'cosine':
        lr_scheduler['base'] = cosine_scheduler(base_value=cfg['hyperparams']['lr']['base_lr'], 
                                        final_value=cfg['hyperparams']['lr']['final_lr'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['lr']['warmup_lr'])
        if cfg['hyperparams']['lr']['predictor_lr'] == 'relative':
            cfg['hyperparams']['lr']['predictor_lr'] = cfg['hyperparams']['lr']['base_lr'] * 10
        lr_scheduler['predictor'] = np.ones((len(train_loader)*cfg['hyperparams']['num_epochs'])) * cfg['hyperparams']['lr']['predictor_lr']
        
    else:
        raise NotImplementedError(print(f"{cfg['hyperparams']['lr']['name']} not implemented"))

    # define wd scheduler
    wd_scheduler = {}
    if cfg['hyperparams']['weight_decay']['name'] == 'cosine':
        
        wd_scheduler['base'] = cosine_scheduler(base_value=cfg['hyperparams']['weight_decay']['base'], 
                                        final_value=cfg['hyperparams']['weight_decay']['final'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['weight_decay']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['weight_decay']['warmup'])
        wd_scheduler['predictor'] =  np.ones((len(train_loader)*cfg['hyperparams']['num_epochs'])) * cfg['hyperparams']['weight_decay']['predictor_wd']
        
    else:
        raise NotImplementedError(print(f"{cfg['hyperparams']['weight_decay']['name']} not implemented"))

    ## try loading from checkpoint
    model, optimizer, start_epoch, amp, rng = create_or_restore_training_state2(args, model, optimizer, logger, amp)

    # Start training
    if 'stop_epoch' in cfg['hyperparams']:
        end_epoch = cfg['hyperparams']['stop_epoch']
    else:
        end_epoch = cfg['hyperparams']['num_epochs']
        
    logger.add_line('='*30 + ' Training Started' + '='*30)

    for epoch in range(start_epoch, end_epoch):
        
        # stopping early during ablation
        if 'stop_epoch' in cfg['hyperparams']:
            if epoch==cfg['hyperparams']['stop_epoch']:
                logger.add_line(f'stopping training using stop_epoch at {epoch}')
                break 
            
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        fwd_kwargs = None
        if 'fwd_kwargs' in cfg['model']:
            fwd_kwargs = cfg['model']['fwd_kwargs']
            fwd_kwargs['distributed'] = args.distributed
        
        train_one_epoch(args, cfg, model, optimizer, 
                        lr_scheduler, wd_scheduler, ema_scheduler,
                        train_loader, 
                        logger, tb_writter, wandb_writter, 
                        epoch, amp, 
                        fwd_kwargs)

        # Save checkpoint
        if args.rank==0:
            logger.add_line('saving model')
            ## normal checkpoint
            commit_state2(args, model, optimizer, epoch, amp, rng, logger)

        # Save just the backbone for further use
        # if args.rank==0 and (epoch+1==end_epoch):
        if args.rank==0 and ((epoch+1==end_epoch) or (epoch+1)%100==0):
            model_path = os.path.join(args.ckpt_dir, f"{cfg['model']['name']}_{args.sub_dir}_{args.db}_ep{epoch}")
            model.module.save_state_dicts(model_path)
            print(f"model is saved to \n{args.ckpt_dir}")            
        
        if args.distributed:
            torch.distributed.barrier() # check this

    # finish logging for this run
    if wandb_writter is not None:
        wandb_writter.finish()
    return

def train_one_epoch(args, cfg, model, optimizer, 
                    lr_scheduler, wd_scheduler, ema_scheduler,
                    train_loader, 
                    logger, tb_writter, wandb_writter, 
                    epoch, amp, fwd_kwargs):
    
    print_freq = cfg['progress']['print_freq']
    model.train()
    batch_size = train_loader.batch_size
    logger.add_line('[Train] Epoch {}'.format(epoch))
    batch_time = AverageMeter('Time', ':6.3f', window_size=100)
    data_time = AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = AverageMeter('Loss', ':.3e')
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_meter, 
                                                 gpu_meter,
                                                 ],
                                          phase='pretext-iter', epoch=epoch, logger=logger, tb_writter=None)
    device = args.gpu if args.gpu is not None else 0
    
    if fwd_kwargs is None:
        fwd_kwargs = {}
    
    end = time.time()
    for i, sample in enumerate(train_loader):
        # break
    
        # update lr & weight decay
        step = epoch * len(train_loader) + i
        for pi, param_group in enumerate(optimizer.param_groups):
            # LR
            if param_group["name"].startswith('base'):
                param_group["lr"] = lr_scheduler['base'][step]
            elif param_group["name"].startswith('predictor'):
                param_group["lr"] = lr_scheduler['predictor'][step]
            else:
                raise ValueError('unknown params')
            # WD
            if wd_scheduler is not None:
                if param_group["name"] in ['base_wd']:
                    param_group["weight_decay"] = wd_scheduler['base'][step]
                elif param_group["name"] in ['predictor_wd']:
                    param_group["weight_decay"] = wd_scheduler['predictor'][step]
           
    
        # measure data loading time
        data_time.update(time.time() - end)
        if train_loader.dataset.return_video:
            # frames = sample['frames'].cuda(device, non_blocking=True)
            frames1 = sample['frames'][:, 0, ::].cuda(device, non_blocking=True)
            frames2 = sample['frames'][:, 1, ::].cuda(device, non_blocking=True)
        
        if train_loader.dataset.return_audio:
            raise NotImplementedError()
            
        optimizer.zero_grad()
        
        if ema_scheduler is not None:
            m = ema_scheduler[step]
        
        if amp is not None: # mixed precision
            with torch.cuda.amp.autocast():
                if train_loader.dataset.return_video:
                    data_dict = model.forward(frames1, frames2, m=m, **fwd_kwargs)
                else:
                    raise ValueError()
        else:
            raise NotImplementedError()
            

        loss = data_dict.pop('loss')
        loss_meter.update(loss, batch_size)
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training") # for log
            logger.add_line(f"Loss is {loss.item()}, stopping training") # for logger
            sys.exit(1)
        
        if amp is not None:
            amp.scale(loss).backward()            
            amp.step(optimizer)
            amp.update()
        else:
            loss.backward()           
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        # other logs
        data_dict.update({'lr':optimizer.param_groups[0]["lr"]})
        data_dict.update({'pred-lr':optimizer.param_groups[2]["lr"]})
        data_dict.update({'wd':optimizer.param_groups[1]["weight_decay"]})
        data_dict.update({'pred-wd':optimizer.param_groups[3]["weight_decay"]})
            
        # measure gpu usage
        gpu_meter.update(torch.cuda.max_memory_allocated()/GB)        

        # print to terminal and tensorboard
        step = epoch * len(train_loader) + i
        if (i+1) % print_freq == 0 or i == 0 or i+1 == len(train_loader):
            progress.display(i+1)

            if tb_writter is not None:
                for kk in data_dict.keys():
                    tb_writter.add_scalar(f'pretext-iter/{kk}', data_dict[kk], step)
                for meter in progress.meters:
                    tb_writter.add_scalar(f'pretext-iter/{meter.name}', meter.val, step)

            if wandb_writter is not None:
                for kk in data_dict.keys():
                    wandb_writter.log({f'pretext-iter/{kk}': data_dict[kk], 'custom_step': step})
                for meter in progress.meters:
                     wandb_writter.log({f'pretext-iter/{meter.name}': meter.val, 'custom_step': step})

        end = time.time()

    # Sync metrics across all GPUs and print final averages
    if args.distributed:
        progress.synchronize_meters(args.gpu)

    if tb_writter is not None:
        tb_writter.add_scalar('pretext-epoch/Epochs', epoch, epoch)
        for meter in progress.meters:
            if meter.name == 'Time' or meter.name == 'Data': # printing total time
                tb_writter.add_scalar(f'pretext-epoch/{meter.name}', meter.sum, epoch)
            else:
                tb_writter.add_scalar(f'pretext-epoch/{meter.name}', meter.avg, epoch)

    if wandb_writter is not None:
        wandb_writter.log({'pretext-epoch/Epochs': epoch, 'custom_step': epoch})
        for meter in progress.meters:
            wandb_writter.log({f'pretext-epoch/{meter.name}': meter.avg, 'custom_step': epoch})

    return


if __name__ == "__main__":

    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)
