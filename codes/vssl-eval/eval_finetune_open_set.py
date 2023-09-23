import math
import sys
import os
import time
import torch
import warnings
import numpy as np
import torch.multiprocessing as mp
import argparse
import yaml
from tools import environment as environ
from tools.utils import sanity_check, set_deterministic
from tools import paths
from checkpointing import create_or_restore_training_state3, commit_state3
from collections import OrderedDict
from datasets.augmentations import get_vid_aug
from datasets import dataloader, FetchSubset  
from models.modules.vit_video import VideoViT
from models import has_batchnorms
from tools import warmup_cosine_scheduler, warmup_multistep_scheduler, set_grad
from tools import synchronize_holder, AverageMeter, ProgressMeter, return_home
from tools.logger import accuracy
from open_set.eval_utils import run_inference, calculate_openness
from open_set.data import get_dataset_original #  in-dist training and val split
from open_set.data import get_osar_val_dataset #  out-dist val split
from open_set.loss import EvidenceLoss
from open_set.head import DebiasHead
from open_set.utils import VidOpenSet, get_params_groups, OSClassifier as Classifier
import pickle


GB = (1024*1024*1024)


def get_args():
        
    parser = argparse.ArgumentParser()
    
    # some stuff
    parser.add_argument("--parent_dir", default="VSSL", help="output folder name",)    
    parser.add_argument("--sub_dir", default="open_set", help="output folder name",)    
    parser.add_argument("--job_id", type=str, default='00', help="jobid=%j")
    parser.add_argument("--server", type=str, default="local", help="location of server",)

    ## debug mode
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=2)
    
    ## dir stuff
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument("--output_dir", default="", help="path where to save")
    parser.add_argument("--resume", default="", help="path where to resume; e.g., 363717")
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
    parser.add_argument("--db", default="kinetics400", help="target db", choices=['ucf101', 'hmdb51', 'kinetics400'])  
    parser.add_argument('-c', '--config-file', type=str, help="pass config file name", 
                        default="byol_ft_k400_hmdb_dear.yaml")
    
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

    open_set_recognition(args, cfg, state, ngpus_per_node, logger=logger, tb_writter=tb_writter, wandb_writter=wandb_writter)
        

def open_set_recognition(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter):
    
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
        
        
    ######################## train data setup
    train_transformations = get_vid_aug(name=cfg['dataset']['train']['vid_transform'],
                                    crop_size=cfg['dataset']['train']['crop_size'],
                                    num_frames=cfg['dataset']['train']['clip_duration']*cfg['dataset']['train']['video_fps'],
                                    mode=cfg['dataset']['train']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'],
                                    )
    
    train_dataset = get_dataset_original(root=paths.my_paths(args.server, 
                                                             cfg['dataset']['train']['name'])[-1], 
                                name=cfg['dataset']['train']['name'],
                                split='train', subset='train',
                                dataset_kwargs=cfg['dataset'],
                                video_transform=train_transformations, 
                                )

    args.effective_batch = cfg['dataset']['train']['batch_size']*args.world_size
    logger.add_line(f"train batch size is {args.effective_batch}")
    logger.add_line(f'Training dataset size: {train_dataset.name} - {len(train_dataset)}') # FIXME
        
    if args.debug:
        train_dataset = FetchSubset(train_dataset, 7)
        

    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset, 
                                              batch_size=cfg['dataset']['train']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)

    ######################## test data setup
    
    val_transformations = get_vid_aug(name=cfg['dataset']['test']['vid_transform'],
                                    crop_size=cfg['dataset']['test']['crop_size'],
                                    num_frames=cfg['dataset']['test']['clip_duration']*cfg['dataset']['test']['video_fps'],
                                    mode=cfg['dataset']['test']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['test']['vid_aug_kwargs'],
                                    )

    # this one will be used for threshold calculation, but using train_transformations that helps in tackling over confident samples
    
    ind_train_dataset = get_dataset_original(root=paths.my_paths(args.server, 
                                                             cfg['dataset']['train']['name'])[-1], 
                                                                name=cfg['dataset']['train']['name'],
                                                                split='train', subset='train',
                                                                dataset_kwargs=cfg['dataset'],
                                                                video_transform=train_transformations,  # change with val if current setup does not work well
                                                                )
    ind_train_dataset.mode = cfg['dataset']['test']['mode'] # load video
    
    logger.add_line("ind_train_dataset")  
    logger.add_line(ind_train_dataset.__repr__())   
    
    
    if args.debug:
        ind_train_dataset = FetchSubset(ind_train_dataset, 17)
        
    ind_train_loader = dataloader.make_dataloader(dataset=ind_train_dataset, 
                                              batch_size=cfg['dataset']['test']['batch_size'],
                                              use_shuffle=False,
                                              drop_last=False,
                                              num_workers=cfg['num_workers']//2,
                                              distributed=args.distributed)
    

    
    ind_val_dataset = get_dataset_original(root=paths.my_paths(args.server, 
                                                      cfg['dataset']['test']['name'])[-1],
                              name=cfg['dataset']['test']['name'],
                              subset='test', split='test',
                              dataset_kwargs=cfg['dataset'],
                              video_transform=val_transformations, 
                              )
    
    logger.add_line("ind_val_dataset")  
    logger.add_line(ind_val_dataset.__repr__())   
    
    if args.debug:
        ind_val_dataset = FetchSubset(ind_val_dataset, 17)
        
    ind_test_loader = dataloader.make_dataloader(dataset=ind_val_dataset, 
                                          batch_size=cfg['dataset']['test']['batch_size'],
                                          use_shuffle=False,
                                          drop_last=False,
                                          num_workers=cfg['num_workers']//2, # to prevent OOM
                                          distributed=args.distributed)
    
    
    ######################## ood test data setup
    
    if cfg['dataset']['ood_type'] == 'new':
        ood_val_dataset = get_osar_val_dataset(root=paths.my_paths(args.server, 
                                                          cfg['dataset']['ood_test']['name'])[-1],
                                  name=cfg['dataset']['ood_test']['name'],
                                  subset=cfg['dataset']['ood_test']['subset'],
                                  dataset_kwargs=cfg['dataset']['test'], # additional kwargs
                                  video_transform=val_transformations, 
                                  )
    elif cfg['dataset']['ood_type'] == 'old':
        
        ood_val_dataset = get_dataset_original(root=paths.my_paths(args.server, 
                                                          cfg['dataset']['ood_test']['name'])[-1],
                                  name=cfg['dataset']['ood_test']['name'],
                                  subset='ood_test', split='test', 
                                  dataset_kwargs=cfg['dataset'],
                                  video_transform=val_transformations, 
                                  )
    
    logger.add_line("ood_val_dataset")  
    logger.add_line(ood_val_dataset.__repr__())   
    
    if args.debug:
        ood_val_dataset = FetchSubset(ood_val_dataset, 17)
        
    ood_test_loader = dataloader.make_dataloader(dataset=ood_val_dataset, 
                                          batch_size=cfg['dataset']['test']['batch_size'],
                                          use_shuffle=False,
                                          drop_last=False,
                                          num_workers=cfg['num_workers']//2, # to prevent OOM
                                          distributed=args.distributed)
    

    ######################## setup model
        
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
    # video_model.load_state_dict(state, strict=True)
    msg = video_model.load_state_dict(state, strict=False)
    print(msg)
    
    logger.add_line(f"{cfg['hyperparams']['criterion']['name']} Loss selected")
    if cfg['hyperparams']['criterion']['name']=='dear':
        criterion = EvidenceLoss(num_classes=cfg['model']['classifier']['num_classes'], 
                                 **cfg['hyperparams']['criterion']['kwargs'])
    elif cfg['hyperparams']['criterion']['name']=='cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    logger.add_line(f"Debias head state: {cfg['model']}")
    if cfg['model']['apply_debias_head']:
        debias_head = DebiasHead(head_in_dim=video_model.embed_dim, 
                             num_classes=cfg['model']['classifier']['num_classes'],
                             **cfg['model']['debias_head'])
    else:
        debias_head = None
        
    classifier = Classifier(feat_dim=video_model.embed_dim, **cfg['model']['classifier'])
    model = VidOpenSet(video_encoder=video_model, 
                    debias_head=debias_head,
                    classifier=classifier,
                    loss=criterion,
                    )
    
    if cfg['model']['vid_setup'] == 'linear':
        set_grad(model.video_encoder, False)
        logger.add_line("Video encoder is set to frozen")
                
    param_groups = get_params_groups(model, 
                        weight_decay=cfg['hyperparams']['optimizer']['weight_decay'], 
                        # no_weight_decay_list=model.video_encoder.video_model.no_weight_decay(), 
                        layer_decay=cfg['hyperparams']['layer_decay'])
        
    
    # assert cfg['model']['text_setup'] == 'frozen', NotImplementedError("text encoder tuning is not supported at the moment")
    
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
            
        
    model, optimizer, start_epoch, amp, rng = create_or_restore_training_state3(args, model, optimizer, logger, amp)
    
    
    ######################## setup training loop
        
    # Start training
    end_epoch = cfg['hyperparams']['num_epochs']
    logger.add_line('='*30 + ' Training Started OpenSet' + '='*30)
    print_freq=cfg['progress']['print_freq']
    fwd_kwargs = cfg['model']['fwd_kwargs']
    close_set_acc, open_set_auc = 0, 0
    best_close_set_acc, best_open_set_auc = 0, 0
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        fwd_kwargs['epoch'] = epoch
        fwd_kwargs['total_epoch'] = end_epoch
        
        marker = time.time()
        # logger.add_line("**************SKIPPING TRAINING IN DEBUG**************")
        train_one_epoch(train_loader, model, optimizer, 
                        lr_scheduler, args, epoch, fwd_kwargs, 
                        logger, tb_writter, wandb_writter, print_freq, amp)
        
        logger.add_line(f'trainng took {time.time()-marker} seconds')
        marker = time.time()
        
        
        if args.rank==0:
            logger.add_line('saving model')    
            commit_state3(args, model, optimizer, epoch, amp, rng, logger)
                
        if args.rank==0:# and ((epoch+1==end_epoch) or (epoch+1)%cfg['eval_freq']==0):
            model_path = os.path.join(args.ckpt_dir, "model.pth.tar")
            torch.save(model.module.state_dict(), model_path)
            print(f"model backbone is saved to \n{args.ckpt_dir}")   
                
                
        logger.add_line(f'model saving took {time.time()-marker} seconds')
        marker = time.time()
        
        # test
        if (epoch+1) % cfg['eval_freq'] == 0 or (epoch+1) == end_epoch:
            
            # ############## calculate threshold
            logger.add_line("calculate threshold...")
            threshold = calculate_uncertainity(model, ind_train_loader, args, cfg, amp, logger, fwd_kwargs)
            logger.add_line(f'The model uncertainty threshold on {train_loader.dataset.name} train set: {threshold}')
            

            # run inference (IND)
            ind_confidences, ind_uncertainties, ind_results, ind_labels = run_inference(model, ind_test_loader, args, cfg, amp, logger, fwd_kwargs)
            # run inference (OOD)
            ood_confidences, ood_uncertainties, ood_results, ood_labels = run_inference(model, ood_test_loader, args, cfg, amp, logger, fwd_kwargs)
            
            ####################### REPORT OPENSET EVAL
            
            results = {'ind_conf':ind_confidences, 'ood_conf':ood_confidences,
                       'ind_unctt':ind_uncertainties, 'ood_unctt':ood_uncertainties, 
                       'ind_pred':ind_results, 'ood_pred':ood_results,
                       'ind_label':ind_labels, 'ood_label':ood_labels}
            
            num_rand = 10  # the number of random selection for ood classes
            ind_ncls = ind_test_loader.dataset.num_classes  # the number of classes in known dataset
            ood_ncls = ood_test_loader.dataset.num_classes  # the number of classes in unknown dataset
            _ood_db_name = ood_test_loader.dataset.name
            _ind_db_name = ind_test_loader.dataset.name
            
            # for thresh in threshold:
            thresh = threshold
            openness_list, macro_F1_list, std_list, close_set_acc, open_set_auc = \
                                                    calculate_openness(results, thresh, 
                                                                   num_rand,
                                                                   ind_ncls,
                                                                   ood_ncls,
                                                                   logger,
                                                                   )
                                                    
                                                    
            logger.add_line(f"openness_list: {openness_list}")
            logger.add_line(f"macro_F1_list: {macro_F1_list}")
            logger.add_line(f"std_list: {std_list}")
            
            # plt.plot(openness_list, macro_F1_list * 100, style, linewidth=2)
            # plt.fill_between(openness_list, macro_F1_list - std_list, macro_F1_list + std_list, style)
                                                                
            if tb_writter is not None:
                tb_writter.add_scalar(f'Eval-{_ind_db_name}-{_ood_db_name}-OpenSet/open_set_auc',open_set_auc, epoch)
                tb_writter.add_scalar(f'Eval-{_ind_db_name}-{_ood_db_name}-OpenSet/close_set_acc',close_set_acc, epoch) # close set is on k400
            
            if best_close_set_acc<close_set_acc:
                best_close_set_acc=close_set_acc
                if args.rank==0:
                    torch.save(model.module.state_dict(), os.path.join(args.ckpt_dir, f"best_model_{_ind_db_name}.pth.tar"))
                    print("model backbone is saved to"+f"{args.ckpt_dir}/"+f"best_model_{_ind_db_name}.pth.tar")   
        
            if best_open_set_auc<open_set_auc: 
                best_open_set_auc=open_set_auc
                if args.rank==0:
                    torch.save(model.module.state_dict(), os.path.join(args.ckpt_dir, f"best_model_{_ood_db_name}.pth.tar"))
                    print("model backbone is saved to"+f"{args.ckpt_dir}/"+f"best_model_{_ood_db_name}.pth.tar")   


    if True:
        _ood_db_name = ood_test_loader.dataset.name
        _ind_db_name = ind_test_loader.dataset.name
        best_path = os.path.join(args.ckpt_dir, f"best_model_{_ood_db_name}.pth.tar")
        state = torch.load(best_path, map_location='cuda:{}'.format(args.gpu))
        model.module.load_state_dict(state, strict=True)
        model.cuda(args.gpu)
        logger.add_line(f"loaded best chk point: {best_path}")
        
        logger.add_line("calculate threshold...")
        threshold = calculate_uncertainity(model, ind_train_loader, args, cfg, amp, logger, fwd_kwargs)
        logger.add_line(f'The model uncertainty threshold on {train_loader.dataset.name} train set: {threshold}')
        
        # run inference (IND)
        ind_confidences, ind_uncertainties, ind_results, ind_labels = run_inference(model, ind_test_loader, args, cfg, amp, logger, fwd_kwargs)
        # run inference (OOD)
        ood_confidences, ood_uncertainties, ood_results, ood_labels = run_inference(model, ood_test_loader, args, cfg, amp, logger, fwd_kwargs)
        
        ####################### REPORT OPENSET EVAL
        
        results = {'ind_conf':ind_confidences, 'ood_conf':ood_confidences,
                   'ind_unctt':ind_uncertainties, 'ood_unctt':ood_uncertainties, 
                   'ind_pred':ind_results, 'ood_pred':ood_results,
                   'ind_label':ind_labels, 'ood_label':ood_labels, 
                   'threshold': threshold}
        
        pickle.dump(results, open(os.path.join(args.log_dir, 'results.pkl'),'wb'))
        
        logger.add_line(f"meta data saved in {os.path.join(args.log_dir, 'results.pkl')}")
    
    return 


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, args, epoch, fwd_kwargs,
                    logger, tb_writter, wandb_writter, print_freq, amp):
    """
    This function is called every epoch to perform training of the network over one full
    (randomized) iteration of the dataset.
    """
    
    model.train()
    # criterion.to(args.gpu)
    
    # class_embedding = train_dataloader.dataset.class_embeddings
    # class_embedding_gpu = torch.from_numpy(class_embedding).to(args.gpu)
    
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
    
    
    
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion.to(args.gpu)
    activation = torch.nn.Softmax(dim=1)
    
    end = time.time()
    for it, sample in enumerate(data_iterator):
        data_time.update(time.time() - end)
        step = epoch * len(train_dataloader) + it
        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)

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
        
        # fwd pass to vision
        optimizer.zero_grad()
        if amp is not None:
            with torch.cuda.amp.autocast():
                loss, logits, all_losses = model.module.forward_train(video, target, **fwd_kwargs)
        else:
            raise NotImplementedError()
            # logits = model(X.to(args.gpu), **fwd_kwargs)


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
        
        # pred_embed = Y.detach().cpu().numpy() # SliceBackward [2, 512]
        # pred_label = cdist(pred_embed, class_embedding, 'cosine').argmin(1)
        # top1 = accuracy_score(l.numpy(), pred_label) * 100
        # top1_meters.update(top1, batch_size)
        # top5_meters.update(top5, batch_size)
        
        # log
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(train_dataloader):
            progress.display(it+1)
            if tb_writter is not None :
                for meter in progress.meters:
                    tb_writter.add_scalar(f'Train-OpenSet-iter/{meter.name}', meter.val, step)
                    
                # all losses for debugging only from rank 0
                for _loss in all_losses:
                    tb_writter.add_scalar(f'Train-OpenSet-losses/{_loss}', all_losses[_loss], step)
            
            if wandb_writter is not None : 
                for meter in progress.meters:
                     wandb_writter.log({f'Train-OpenSet-iter/{meter.name}': meter.val, 'custom_step': step})
    

    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(train_dataloader) * args.world_size)
        
    logger.add_line(f'Train-Epoch: {epoch} - {name}_OpenSet/top1: {top1_meters.avg}')
    logger.add_line(f'Train-Epoch: {epoch} - {name}_OpenSet/top5: {top5_meters.avg}')
    
    if tb_writter is not None :
        tb_writter.add_scalar('Train-OpenSet/top1', top1_meters.avg, epoch)
        tb_writter.add_scalar('Train-OpenSet/top5', top5_meters.avg, epoch)
    if wandb_writter is not None : 
        wandb_writter.log({'Train-OpenSet/top1': top1_meters.avg, 'custom_step': epoch})
        wandb_writter.log({'Train-OpenSet/top5': top5_meters.avg, 'custom_step': epoch})
    
    torch.cuda.empty_cache()

    return 


  
    

def calculate_uncertainity(model, train_loader, args, cfg, amp, logger, fwd_kwargs):
    
    ######################### CALCULATE THRESOLD USING IND TRAIN SET #########################
    
    # using training set (IND)
    _, all_uncertainties, _, _ = run_inference(model, train_loader, args, cfg, amp, logger, fwd_kwargs)
    uncertain_sort = np.sort(all_uncertainties)[::-1]  # sort the uncertainties with descending order
    N = all_uncertainties.shape[0]
    topK = N - int(N * 0.95)
    threshold = uncertain_sort[topK-1]
    
    return threshold
    
    
if __name__ == "__main__":
    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)

