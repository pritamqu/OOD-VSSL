import math
import sys
import os
import time
import torch
import warnings
import yaml
import numpy as np
import torch.multiprocessing as mp
import argparse
from tools import environment as environ
from tools.utils import sanity_check, set_deterministic
from tools import paths
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from collections import defaultdict
from checkpointing import create_or_restore_training_state3, commit_state3
from datasets.augmentations import get_vid_aug
from datasets import dataloader, FetchSubset  
from zsl.data import get_dataset
from zsl.word2vec import load_word2vec
from zsl.utils import MLP, VidText, get_params_groups
from models.modules.vit_video import VideoViT
from models import has_batchnorms
from tools import warmup_cosine_scheduler, warmup_multistep_scheduler, set_grad
from tools import AverageMeter, ProgressMeter, return_home
from tools.logger import accuracy
from collections import OrderedDict

""" training algo is based on 
https://arxiv.org/pdf/2003.01455.pdf

"""

GB = (1024*1024*1024)

def get_args():
        
    parser = argparse.ArgumentParser()
    
    # some stuff
    parser.add_argument("--parent_dir", default="VSSL", help="output folder name",)    
    parser.add_argument("--sub_dir", default="zsl", help="output folder name",)    
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
    parser.add_argument("--db", default="kinetics400", help="target db", choices=['kinetics400', 'kinetics700'])  
    parser.add_argument('-c', '--config-file', type=str, help="pass config file name", 
                        default="byol_zsl.yaml")
    
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
            # Map model to be loaded to specified single gpu.
            state = torch.load(path, map_location='cuda:{}'.format(args.gpu))
    else:
        raise FileNotFoundError (f'weight is not found at {path}')

    zero_shot_recognition(args, cfg, state, ngpus_per_node, logger=logger, tb_writter=tb_writter, wandb_writter=wandb_writter)
        

def zero_shot_recognition(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter):
    
    if args.distributed and args.gpu is not None:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        batch_size = cfg['dataset']['train']['batch_size']
        batch_size = int(batch_size / ngpus_per_node)
        cfg['num_workers'] = int((cfg['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)
        cfg['dataset']['train']['batch_size'] = batch_size
        
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
        cfg['eval_freq'] = 1
        if args.server == 'local':
            cfg['dataset']['train']['batch_size'] = 2
            cfg['hyperparams']['num_epochs'] = 3
        
        
    ######################## setup data
    if cfg['text_model'] in ['Word2Vec', 'word2vec']:
        text_model = load_word2vec(return_home(args.server)+'/ASSETS/Word2Vec')
        from zsl.word2vec import classes2embedding
    else:
        raise NotImplementedError(f"{cfg['text_model']}")
        
        
    train_transformations = get_vid_aug(name=cfg['dataset']['train']['vid_transform'],
                                    crop_size=cfg['dataset']['train']['crop_size'],
                                    num_frames=cfg['dataset']['train']['clip_duration']*cfg['dataset']['train']['video_fps'],
                                    mode=cfg['dataset']['train']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'],
                                    )
    
    train_dataset = get_dataset(root=paths.my_paths(args.server, cfg['dataset']['train']['name'])[-1], 
                                name=cfg['dataset']['train']['name'],
                                subset=cfg['dataset']['train']['split'],
                                dataset_kwargs=cfg['dataset']['train'],
                                video_transform=train_transformations, 
                                wv_model = text_model,
                                classes2embedding = classes2embedding,
                                )
    
    val_transformations = get_vid_aug(name=cfg['dataset']['test']['vid_transform'],
                                    crop_size=cfg['dataset']['test']['crop_size'],
                                    num_frames=cfg['dataset']['test']['clip_duration']*cfg['dataset']['test']['video_fps'],
                                    mode=cfg['dataset']['test']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['test']['vid_aug_kwargs'])  
    
    train_db_name = cfg['dataset']['train']['name']
    val_db_names = cfg['dataset']['test']['name']
    subsets = cfg['dataset']['test']['split']
    assert len(val_db_names) == len(subsets)
    if not isinstance(val_db_names, list):
        val_db_names = [val_db_names]
        subsets = [subsets]
    val_dataset = []
    for kk in range(len(val_db_names)):
        val_dataset.append(get_dataset(root=paths.my_paths(args.server, val_db_names[kk])[-1],
                                  name=val_db_names[kk],
                                  subset=cfg['dataset']['test']['split'][kk],
                                  dataset_kwargs=cfg['dataset']['test'],
                                  video_transform=val_transformations, 
                                  wv_model = text_model,
                                  classes2embedding = classes2embedding,
                                  )
                           )

    args.effective_batch = cfg['dataset']['train']['batch_size']*args.world_size
    logger.add_line(f"train batch size is {args.effective_batch}")
    logger.add_line(f'Training dataset size: {train_dataset.name} - {len(train_dataset)}') # FIXME
    for _vd in val_dataset:
        logger.add_line(f'Validation dataset size: {_vd.name} - {len(_vd)}')
       
        
    if args.debug:
        train_dataset = FetchSubset(train_dataset, 7)
        _val_dataset = []
        for k in val_dataset:
            _val_dataset.append(FetchSubset(k, 5))
        cfg['dataset']['test']['batch_size'] = 2
        val_dataset = _val_dataset
        

    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset, 
                                              batch_size=cfg['dataset']['train']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)
    test_loader = []
    for kk in range(len(val_db_names)):
        test_loader.append(dataloader.make_dataloader(dataset=val_dataset[kk], 
                                          batch_size=cfg['dataset']['test']['batch_size'],
                                          use_shuffle=cfg['dataset']['test']['use_shuffle'],
                                          drop_last=cfg['dataset']['test']['drop_last'],
                                          num_workers=cfg['num_workers']//2, # to prevent OOM
                                          distributed=args.distributed)
                           )
    
    
    ######################## setup model
    
    """ the video backbone will be trained the text setup will remain same 
    can try training text encoder as well
    """
    
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
    video_model.load_state_dict(state, strict=True)
    
    if cfg['model']['classifier'] == 'default':
        classifier = cfg['model']['classifier']
    else:
        raise NotImplementedError()
    text_encoder = MLP(**cfg['model']['text_encoder'])
    model = VidText(video_encoder=video_model, 
                    text_encoder=text_encoder,
                    classifier=classifier 
                    )

    if cfg['model']['vid_setup'] == 'linear':
        set_grad(model.video_encoder, False)
        logger.add_line("Video encoder is made frozen")
                
    param_groups = get_params_groups(model, 
                        weight_decay=cfg['hyperparams']['optimizer']['weight_decay'], 
                        # no_weight_decay_list=model.video_encoder.video_model.no_weight_decay(), 
                        layer_decay=cfg['hyperparams']['layer_decay'])
        
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.add_line('number of params (M): %.2f' % (n_parameters / 1.e6))
    # sync bn if used
    if args.distributed and cfg['sync_bn']:
        if has_batchnorms(model):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # distribute the model
    model = model.cuda(args.gpu)
    model = environ.distribute_model(models=model, 
                                           args=args, ngpus_per_node=ngpus_per_node)
    
    
    ######################## setup model params and others

    # optimizer
    if cfg['hyperparams']['optimizer']['name']=='sgd':
        optimizer = torch.optim.SGD(param_groups, 
                                    lr=1e-3, # setting lr through lr scheduler
                                    momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adam':
        optimizer = torch.optim.Adam(param_groups, 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adamw':
        optimizer = torch.optim.AdamW(param_groups, 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
            
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
            
    if cfg['hyperparams']['criterion']=='mse':
        criterion = torch.nn.MSELoss()
    elif cfg['hyperparams']['criterion']=='cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()
    
    model, optimizer, start_epoch, amp, rng = create_or_restore_training_state3(args, model, optimizer, logger, amp)
    
    
    ######################## setup training loop
        
    # Start training
    end_epoch = cfg['hyperparams']['num_epochs']
    logger.add_line('='*30 + ' Training Started ZSL' + '='*30)
    print_freq=cfg['progress']['print_freq']
    fwd_kwargs = cfg['model']['fwd_kwargs']
    result_holder = defaultdict(list)
    best_result_holder = defaultdict(list)
    val_db_names = [_dataloader.dataset.name for _dataloader in test_loader[:1]]
    for val_db_name in val_db_names:
        result_holder[val_db_name] = []
            
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        
        
        marker = time.time()
        train_one_epoch(train_loader, model, optimizer, criterion, 
                        lr_scheduler, args, epoch, fwd_kwargs, 
                        logger, tb_writter, wandb_writter, print_freq, amp)
        
        logger.add_line(f'trainng took {time.time()-marker} seconds')
        marker = time.time()
        
        if args.rank==0:
            logger.add_line('saving model')    
            commit_state3(args, model, optimizer, epoch, amp, rng, logger)
                
        if args.rank==0:
            model_path = os.path.join(args.ckpt_dir, "model.pth.tar")
            torch.save(model.module.state_dict(), model_path)
            print(f"model backbone is saved to \n{args.ckpt_dir}")   
                
        logger.add_line(f'model saving took {time.time()-marker} seconds')
        marker = time.time()
        
        # InD test
        if (epoch+1) % cfg['eval_freq'] == 0 or (epoch+1) == end_epoch:
            for _loader in test_loader[:1]: # first one is InD val and there is only one InD val to track
                val_db_name = _loader.dataset.name
                assert val_db_name.startswith('Kinetics') # sanity
                logger.add_line(f'Evaluating on {val_db_name}')
                result_holder[val_db_name]=[
                evaluate(_loader, tb_writter, epoch, model, args, fwd_kwargs, logger, tb_writter, 
                         wandb_writter, print_freq, amp)
                    ]
                
            if len(best_result_holder)==0:
                best_result_holder=result_holder
            else:
                for _db_name in best_result_holder:
                    if best_result_holder[_db_name][0][0]<=result_holder[_db_name][0][0]: # acc_top1
                        best_result_holder[_db_name]=result_holder[_db_name]
                        if args.rank==0 and _db_name.startswith('Kinetics'):
                            best_ind_path = os.path.join(args.ckpt_dir, f"best_model_{_db_name}.pth.tar")
                            torch.save(model.module.state_dict(), best_ind_path)
                            print("model backbone is saved to"+f"{args.ckpt_dir}/"+f"best_model_{_db_name}.pth.tar")   
                
    # ind and ood test at the end using the best model
    # load the best ind model
    # best_ind_path=os.path.join(args.ckpt_dir, f"best_model_{train_db_name}.pth.tar")
    best_state = torch.load(best_ind_path, map_location='cuda:{}'.format(args.gpu))
    model.module.load_state_dict(best_state)

    for _loader in test_loader: # run through all val
        _db_name = _loader.dataset.name
        logger.add_line(f'Evaluating on {_db_name}')
        top1, top5 = evaluate(_loader, tb_writter, epoch, model, args, fwd_kwargs, logger, tb_writter, 
                    wandb_writter, print_freq, amp)

        logger.add_line(f"Best result: {_db_name} - top1 {top1} - top5 {top5}")
        if tb_writter is not None:
            tb_writter.add_scalar('Eval-Best-{kk}-ZSL/top1', top1, epoch)
            tb_writter.add_scalar('Eval-Best-{kk}-ZSL/top5', top5, epoch)
        if wandb_writter is not None:
            wandb_writter.log({f'Eval-Best-{kk}-ZSL/top1': top1, 'custom_step': epoch})
            wandb_writter.log({f'Eval-Best-{kk}-ZSL/top5': top5, 'custom_step': epoch})
            
    return 


def train_one_epoch(train_dataloader, model, optimizer, criterion, lr_scheduler, opt, epoch, fwd_kwargs,
                    logger, tb_writter, wandb_writter, print_freq, amp):
    """
    This function is called every epoch to perform training of the network over one full
    (randomized) iteration of the dataset.
    """
    
    model.train()
    criterion.to(opt.gpu)
    
    class_embedding = train_dataloader.dataset.class_embeddings
    class_embedding_gpu = torch.from_numpy(class_embedding).to(opt.gpu)
    
    # class_names = train_dataloader.dataset.classes
    batch_size = train_dataloader.batch_size
    name = train_dataloader.dataset.name
    
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    max_lr_meter = AverageMeter('Max_LR', ':.4e', 0)
    min_lr_meter = AverageMeter('Min_LR', ':.4e', 0)
    weight_decay_meter = AverageMeter('WD', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    
    progress = ProgressMeter(len(train_dataloader), meters=[batch_time, data_time, loss_meters, 
                                                      max_lr_meter, min_lr_meter, weight_decay_meter,
                                                      top1_meters, top5_meters, gpu_meter,], phase='Train', epoch=epoch, logger=logger)
    
    data_iterator = train_dataloader
    
    
    
    criterion = torch.nn.CrossEntropyLoss()
    activation = torch.nn.Softmax(dim=1)
    
    end = time.time()
    for it, sample in enumerate(data_iterator):
        data_time.update(time.time() - end)
        step = epoch * len(train_dataloader) + it
        X, Z = sample['frames'], sample['class_embeddings']
        target = sample['label'].cuda()
        
        # update LR
        min_lr = 10.
        max_lr = 0.
        weight_decay_value = 0.
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr_scheduler[step] * param_group["lr_scale"]
            else:
                param_group["lr"] = lr_scheduler[step]
            
            min_lr = min(min_lr, param_group["lr"])
            max_lr = max(max_lr, param_group["lr"])
            
            if param_group["weight_decay"] > 0:
                weight_decay_value = param_group["weight_decay"]
                    
        max_lr_meter.update(max_lr, batch_size)
        min_lr_meter.update(min_lr, batch_size)
        weight_decay_meter.update(weight_decay_value, batch_size)
            
            
        # batch_times.append(time.time() - tt_batch)
        # s = list(X.shape)
        X = X.to(opt.gpu)
        # Z = Z.to(opt.gpu)
        
        
        
        # fwd pass to vision
        optimizer.zero_grad()
        if amp is not None:
            with torch.cuda.amp.autocast():
                logits = model(X, class_embedding_gpu, **fwd_kwargs)
        else:
            raise NotImplementedError()
            # logits = model(X.to(opt.gpu), **fwd_kwargs)

        loss = criterion(logits, target)

        # # Compute loss.
        # loss = criterion(Y, Z)
        
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
        
        
        #Store loss per iteration.
        loss_meters.update(loss.item(), batch_size)
        gpu_meter.update(torch.cuda.max_memory_allocated()/GB) 
        batch_time.update(time.time() - end)
        end = time.time()

        # Compute Accuracy.
        
        confidence = activation(logits)
        acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
        top1_meters.update(acc1[0].item(), target.size(0))
        top5_meters.update(acc5[0].item(), target.size(0))
                
        # log
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(train_dataloader):
            progress.display(it+1)
            if tb_writter is not None :
                for meter in progress.meters:
                    tb_writter.add_scalar(f'Train-ZSL-iter/{meter.name}', meter.val, step)
            
            if wandb_writter is not None : 
                for meter in progress.meters:
                     wandb_writter.log({f'Train-ZSL-iter/{meter.name}': meter.val, 'custom_step': step})
    

    if opt.distributed:
        progress.synchronize_meters_custom(opt.gpu)
        progress.display(len(train_dataloader) * opt.world_size)
        
    logger.add_line(f'Train-Epoch: {epoch} - {name}_ZSL/top1: {top1_meters.avg}')
    # logger.add_line(f'Train-Epoch: {epoch} - {name}_ZSL/top5: {top5_meters.avg}')
    
    if tb_writter is not None :
        tb_writter.add_scalar('Train-ZSL/top1', top1_meters.avg, epoch)
        # tb_writter.add_scalar('Train-ZSL/top5', top5_meters.avg, epoch)
    if wandb_writter is not None : 
        wandb_writter.log({'Train-ZSL/top1': top1_meters.avg, 'custom_step': epoch})
        # wandb_writter.log({'Train-ZSL/top5': top5_meters.avg, 'custom_step': epoch})
    
    torch.cuda.empty_cache()

    return 



def evaluate(test_dataloader, txwriter, epoch, model, opt, fwd_kwargs, logger, tb_writter, 
             wandb_writter, print_freq, amp):
    """
    This function is called in the end to evaluate the model on 50% of the classes.
    """
    
    model.eval()
    name = test_dataloader.dataset.name
    is_video = True if test_dataloader.dataset.mode == 'video' else False
    class_embedding = test_dataloader.dataset.class_embeddings
    class_embedding_gpu = torch.from_numpy(class_embedding).to(opt.gpu)
    criterion = torch.nn.CrossEntropyLoss()
    activation = torch.nn.Softmax(dim=1)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    
    progress = ProgressMeter(len(test_dataloader), meters=[loss_meters, top1_meters, top5_meters,
                                                  ], phase=f"Eval-{name}-ZSL", epoch=epoch, logger=logger)
    
    with torch.no_grad():
        
        for idx, sample in enumerate(test_dataloader):
            X, Z = sample['frames'], sample['class_embeddings']
            target = sample['label'].cuda()
                        
            # tackle video
            if is_video: # test_dense
                batch_size, clips_per_sample = X.shape[0], X.shape[1]
                X = X.flatten(0, 1).contiguous()
            
            X = X.to(opt.gpu)
            
            
            # Run network on batch
            with torch.no_grad():
                if amp is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(X, class_embedding_gpu, **fwd_kwargs)
                else:
                    raise NotImplementedError()
                    # Y = model(X.to(opt.gpu), **fwd_kwargs)
            
            if is_video:
                confidence = activation(logits).view(batch_size, clips_per_sample, -1).mean(1)
                target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
                loss = criterion(logits, target_tiled)
            else:
                confidence = activation(logits)
                loss = criterion(logits, target)
                
            acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
            top1_meters.update(acc1[0].item(), target.size(0))
            top5_meters.update(acc5[0].item(), target.size(0))
            
            logger.add_line(f"Eval on {test_dataloader.dataset.name} - {idx}/{len(test_dataloader)}")
            

    
    if opt.distributed:
        # meters
        progress.synchronize_meters_custom(opt.gpu)
        progress.display(len(test_dataloader) * opt.world_size)
        
    
    accuracy_top1, accuracy_top5 = top1_meters.avg, top5_meters.avg
    
    logger.add_line(f'Epoch: {epoch} - {name}_ZSL/top1: {accuracy_top1}')
    logger.add_line(f'Epoch: {epoch} - {name}_ZSL/top5: {accuracy_top5}')
    
    if tb_writter is not None :
        tb_writter.add_scalar(f'Eval-{name}-ZSL/top1', accuracy_top1, epoch)
        tb_writter.add_scalar(f'Eval-{name}-ZSL/top5', accuracy_top5, epoch)
    if wandb_writter is not None : 
        wandb_writter.log({f'Eval-{name}-ZSL/top1': accuracy_top1, 'custom_step': epoch})
        wandb_writter.log({f'Eval-{name}-ZSL/top5': accuracy_top5, 'custom_step': epoch})

    
    torch.cuda.empty_cache()

    return accuracy_top1, accuracy_top5
    
def compute_accuracy(predicted_embed, class_embed, true_embed):
    """
    Compute accuracy based on the closest Word2Vec class
    """
    assert len(predicted_embed) == len(true_embed), "True and predicted labels must have the same number of samples"
    y_pred = cdist(predicted_embed, class_embed, 'cosine').argsort(1)
    y = cdist(true_embed, class_embed, 'cosine').argmin(1)
    accuracy = accuracy_score(y, y_pred[:, 0]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, :5])]) * 100
    return accuracy, accuracy_top5



if __name__ == "__main__":
    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)
