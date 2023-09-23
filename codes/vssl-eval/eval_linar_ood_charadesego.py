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
import math
import sys
from tools import AverageMeter, ProgressMeter, accuracy, warmup_cosine_scheduler, warmup_multistep_scheduler, set_grad
from datasets.augmentations import get_vid_aug
from datasets import get_dataset, dataloader, FetchSubset        
from checkpointing import create_or_restore_training_state3, commit_state3
from models import has_batchnorms
from models import VideoViT
from collections import OrderedDict
from tools import paths
from tools import return_home
from models.modules.vid_text import NormSoftmaxLoss, VidText, get_params_groups, AllGather_multi, sim_matrix
from transformers import AutoModel, AutoTokenizer
from csv import reader
GB = (1024*1024*1024)

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
    parser.add_argument("--db", default="charadesego", help="target db", choices=['charadesego'])  
    parser.add_argument('-c', '--config-file', type=str, help="pass config file name", 
                        default="byol_train3rd_test1st_infonce_linear_v1.yaml")

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
    video_model=VideoViT(**cfg['model']['vid_backbone'])    
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
    msg = video_model.load_state_dict(state, strict=False)
    print(msg)

    text_encoder = AutoModel.from_pretrained('distilbert-base-uncased',
                   cache_dir=return_home(args.server)+'/ASSETS/distilbert-base-uncased')
    # if you encounter error here; that means this model is missing, although it should be able to download
    # you can also download the model from here: https://drive.google.com/file/d/1h5TSb30pr5Ah53xUBLCoWOGRaJhvcy0h/view?usp=sharing
    # unzip and place it in the same path
    model = VidText(video_encoder=video_model, 
                    text_encoder=text_encoder,
                    )
    
    if cfg['model']['vid_setup']=='linear':
        set_grad(model.video_encoder, False)
        logger.add_line("Video encoder is set to frozen")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", 
                   cache_dir=return_home(args.server)+'/ASSETS/distilbert-base-uncased',
                                          TOKENIZERS_PARALLELISM=False)

    # check no. of model params
    param_groups = get_params_groups(model, 
                        weight_decay=cfg['hyperparams']['weight_decay'], 
                        layer_decay=cfg['hyperparams']['layer_decay'])
    
    
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


        
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=train_transformations, 
                                split='train')

    
        
    val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=val_transformations, 
                                split='test')
    
    ########### setup ood dataset
    assert cfg['dataset']['ood_test']['name'] == 'charadesego_text'
    ood_db = cfg['dataset']['ood_test']['name']
    ind_db = cfg['dataset']['name']
    ood_val_dataset = get_dataset(root=paths.my_paths(args.server, args.db)[-1],
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
    
                
    # adjusting batch size as test is done in dense mode
    test_batch_size = max(cfg['dataset']['batch_size'] // (3*cfg['dataset']['test']['clips_per_video']), 1)

    if args.debug:
        train_dataset = FetchSubset(train_dataset, 17)
        val_dataset = FetchSubset(val_dataset, 11)
        ood_val_dataset = FetchSubset(ood_val_dataset, 11)
        test_batch_size = 2
        
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
    
    criterion = NormSoftmaxLoss()
        
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
        
    best_top1, best_top5=0, 0
    best_epoch=start_epoch
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        fwd_kwargs = cfg['model']['fwd_kwargs']
        
        # # train
        marker = time.time()
        
        #------ lazy to change here mean_ap-> top1; mean_wap-> top5
        
        # results['mean_auc'], results['mean_ap'], results['mean_wap']
        train_one_epoch(phase='train', 
                                      loader=train_loader, 
                                      model=model,
                                      tokenizer=tokenizer,
                                      optimizer=optimizer, 
                                      criterion=criterion, 
                                      # mixup_fn=mixup_fn,
                                      fwd_kwargs=fwd_kwargs,
                                      lr_scheduler=lr_scheduler, 
                                      # wd_scheduler=None, 
                                      amp=amp,
                                      epoch=epoch, 
                                      args=args, logger=logger, 
                                      tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                      print_freq=cfg['progress']['print_freq'], 
                                      )     
        
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
        
        # test
        if (epoch+1) % cfg['eval_freq'] == 0 or (epoch+1) == end_epoch:
            # results['mean_auc'], results['mean_ap'], results['mean_wap']
            vid_top1 = eval_one_epoch(phase='test_dense', 
                                          loader=test_loader, 
                                          model=model,
                                          tokenizer=tokenizer,
                                          criterion=criterion, 
                                          fwd_kwargs=fwd_kwargs,
                                          amp=amp,
                                          epoch=epoch, 
                                          args=args, logger=logger, 
                                          tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                          print_freq=cfg['progress']['print_freq'], 
                                          )

            logger.add_line(f'validation took {time.time()-marker} seconds')
            logger.add_line(f'InD Validation at Epoch: {epoch+1} meanAP: {vid_top1}')
            top1 = vid_top1
            if top1>best_top1:
                best_top1=top1
                best_epoch=epoch
                # Save checkpoint
                best_ind_model=os.path.join(args.ckpt_dir, f"best_model_{ind_db}.pth.tar")
                if args.rank==0:
                    torch.save(model.module.state_dict(), best_ind_model)
                
            if tb_writter is not None:
                tb_writter.add_scalar('fine_tune_epoch/te_mean_ap', vid_top1, epoch)
            if wandb_writter is not None:
                wandb_writter.log({'fine_tune_epoch/te_mean_ap': vid_top1, 'custom_step': epoch})
                
                
    # ood test at the end
    # load the best ind model
    best_state = torch.load(best_ind_model, map_location='cuda:{}'.format(args.gpu))
    model.module.load_state_dict(best_state)
    ood_vid_top1 = eval_one_epoch(phase='test_dense', 
                                    loader=ood_test_loader, 
                                    model=model,
                                    tokenizer=tokenizer,
                                    criterion=criterion, 
                                    fwd_kwargs=fwd_kwargs,
                                    amp=amp,
                                    epoch=epoch, 
                                    args=args, logger=logger, 
                                    tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                    print_freq=cfg['progress']['print_freq'], 
                                    )

    logger.add_line(f'ood validation took {time.time()-marker} seconds')

    if tb_writter is not None:
        tb_writter.add_scalar('fine_tune_epoch/ood_mean_ap', ood_vid_top1, epoch)
    if wandb_writter is not None:
        wandb_writter.log({'fine_tune_epoch/ood_mean_ap': ood_vid_top1, 'custom_step': epoch})
                
                  
    # --------- end log               
    if args.rank==0:
        logger.add_line(f'InD meanAP: {best_top1}')
        logger.add_line(f'OOD meanAP: {ood_vid_top1}')

    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()
        
    return



def train_one_epoch(phase, loader, model, tokenizer, optimizer, criterion, fwd_kwargs,
              lr_scheduler, amp,
              epoch, args, logger, tb_writter, wandb_writter, print_freq,):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    max_lr_meter = AverageMeter('Max_LR', ':.4e', 0)
    min_lr_meter = AverageMeter('Min_LR', ':.4e', 0)
    weight_decay_meter = AverageMeter('WD', ':.4e', 0)
    # top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    # top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, 
                                                  max_lr_meter, min_lr_meter, weight_decay_meter,
                                                  # top1_meters, top5_meters, 
                                                  gpu_meter,], phase=phase, epoch=epoch, logger=logger)

    model.train()
    allgather = AllGather_multi()
    criterion = criterion.to(args.gpu)
    n_gpu = args.world_size
    end = time.time()
    results = None
    
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        
        step = epoch * len(loader) + it           
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr_scheduler[step] * param_group["lr_scale"]
            else:
                param_group["lr"] = lr_scheduler[step]
            

        # prepare data
        video = sample['frames']
        text = sample['text']
        text = tokenizer(sample['text'], return_tensors='pt', padding=True,
                                                  truncation=True)
        
        text = {key: val.to(args.gpu) for key, val in text.items()}
        video = video.cuda(args.gpu, non_blocking=True)
        target = sample['label'].cuda()
        
        optimizer.zero_grad()
        if amp is not None:
            with torch.cuda.amp.autocast():
                text_embeds, video_embeds = model(video, text, **fwd_kwargs)
                if args.distributed:
                    video_embeds = allgather.apply(video_embeds, n_gpu, args)
                    text_embeds = allgather.apply(text_embeds, n_gpu, args)
                output = sim_matrix(text_embeds, video_embeds)
                loss = criterion(output)
        else:
            raise NotImplementedError()
        
        if amp is not None:
            amp.scale(loss).backward()
            amp.step(optimizer)
            amp.update()
        else:
            loss.backward()
            optimizer.step()

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training") # for log
            logger.add_line(f"Loss is {loss.item()}, stopping training") # for logger
            sys.exit(1)
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        gpu_meter.update(torch.cuda.max_memory_allocated()/GB) 
        loss_meters.update(loss.item(), target.size(0))
           
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
                for meter in progress.meters:
                    tb_writter.add_scalar(f'{phase}_fine_tune_iter/{meter.name}', meter.val, step)
            
            if wandb_writter is not None : 
                for meter in progress.meters:
                     wandb_writter.log({f'{phase}_fine_tune_iter/{meter.name}': meter.val, 'custom_step': step})
            
    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    return 


def eval_one_epoch(phase, loader, model, tokenizer, criterion, fwd_kwargs,
              amp, epoch, args, logger, tb_writter, wandb_writter, print_freq,):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    # top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    # top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, 
                                                  gpu_meter,], phase=phase, epoch=epoch, logger=logger)

    model.eval()
    allgather = AllGather_multi()
    criterion = criterion.to(args.gpu)
    n_gpu = args.world_size
    end = time.time()
    is_video = True if loader.dataset.mode == 'video' else False
    vid_embed_arr = []
    target_arr = []
    with torch.no_grad():
        
        # construct set of sentences.
        cls_arr = []
        cls_file = os.path.dirname(loader.dataset.meta_dir) + '/Charades_v1_classes.txt'
        with open(cls_file, 'r') as charades:
            csv_reader = list(reader(charades))
        for line in csv_reader:
            cls_arr.append(line[0][5:])

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        data_cls = tokenizer(cls_arr, return_tensors='pt', padding=True, truncation=True)
        data_cls = {key: val.cuda() for key, val in data_cls.items()}
        
        text_embed = model.module.forward_text(data_cls)
        text_embeds = text_embed.cpu().detach() # embeds of all classes
        
    
        for it, sample in enumerate(loader):
            data_time.update(time.time() - end)
            
            step = epoch * len(loader) + it           
            # prepare data
            video = sample['frames']
            text = sample['text']
            text = tokenizer(sample['text'], return_tensors='pt', padding=True,
                                                      truncation=True)
            
            text = {key: val.to(args.gpu) for key, val in text.items()}
            video = video.cuda(args.gpu, non_blocking=True)
            target = sample['label'].cuda()
            
            if is_video:
                batch_size, clips_per_sample = video.shape[0], video.shape[1]
                video = video.flatten(0, 1).contiguous()
            
            # optimizer.zero_grad()
            # if amp is not None:
                # with torch.cuda.amp.autocast():
            video_embeds = model.module.forward_video(video, **fwd_kwargs)
            if is_video:
                video_embeds = video_embeds.view(batch_size, clips_per_sample, -1).mean(1)
            if args.distributed:
                vid_embed_all = [torch.zeros_like(video_embeds) for _ in range(n_gpu)]
                torch.distributed.all_gather(vid_embed_all, video_embeds)
                vid_embed_all = torch.cat(vid_embed_all, dim=0)
                vid_embed_arr.append(vid_embed_all.cpu())

                data_target_all = [torch.zeros_like(target) for _ in range(n_gpu)]
                torch.distributed.all_gather(data_target_all, target)
                data_target_all = torch.cat(data_target_all, dim=0)
                target_arr.append(data_target_all.cpu())
            else:
                vid_embed_arr.append(video_embeds.cpu())
                target_arr.append(target.cpu())

        # calc
        vid_embeds = torch.cat(vid_embed_arr)
        target_embeds = torch.cat(target_arr)

        sims = sim_matrix(text_embeds, vid_embeds).numpy().T
        targets = target_embeds.numpy()
        
        results = charades_metrics(sims, targets)
                
                
    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    return results['mean_ap']


def charades_metrics(submission_array, gt_array):
    # https://github.com/showlab/EgoVLP/blob/main/model/metric.py
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    metrics = {}
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    m_ap, w_ap, m_aps = mean_ap(fix, gt_array)
    metrics['mean_ap'] = m_ap
    # metrics['wAP'] = w_ap
    # metrics['mAPs'] = m_aps
    return metrics

def mean_ap(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps

if __name__ == "__main__":
    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)

