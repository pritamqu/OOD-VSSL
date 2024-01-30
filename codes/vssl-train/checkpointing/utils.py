import torch
import os 
from datetime import datetime

def create_or_restore_training_state2(args, model, optimizer, logger, amp):
    
    def _system_restore(training_state, model, optimizer, amp):
        model.load_state_dict(training_state['model'])
        optimizer.load_state_dict(training_state['optimizer'])
        start_epoch = training_state['epoch']+1
        rng = training_state['rng'].cpu()
        torch.random.set_rng_state(rng)
        if amp is not None:
                amp.load_state_dict(training_state['amp'])
        
        return model, optimizer, start_epoch, amp, rng
    
    logger.add_line("looking for checkpoint restoration...")    
    path = None
    start_epoch=0
    rng = torch.random.get_rng_state()
    if os.path.isfile(os.path.join(args.checkpoint_path, 'checkpoint.pth.tar')):
        path = os.path.join(args.checkpoint_path, 'checkpoint.pth.tar')
    if args.resume and os.path.isfile(os.path.join(args.resume, 'checkpoint.pth.tar')):
        path = os.path.join(args.resume, 'checkpoint.pth.tar')
        
    if path is not None:
        if args.gpu is None:
            training_state = torch.load(path)
        else:
            # Map model to be loaded to specified single gpu.
            training_state = torch.load(path, map_location='cuda:{}'.format(args.gpu))
        model, optimizer, start_epoch, amp, rng = _system_restore(training_state, 
                                                                           model, optimizer, amp)

        logger.add_line(f"training state restored from {path} at epoch: {start_epoch}")
        
    else:
        logger.add_line(f"No checkpoint found at {path} so no chkpt either at {args.resume} or {args.checkpoint_path}")

    return model, optimizer, start_epoch, amp, rng


# we need to be careful when saving checkpoints since preemption can also
# occur during checkpointing. Therefore, we need to make sure the checkpoint
# file is either kept untouched or successfully updated during this process.
def commit_state2(args, model, optimizer, epoch, amp, rng, logger):
    
    # for system restoration
    checkpoint_path = os.path.join(args.checkpoint_path, 'checkpoint.pth.tar')
    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pth.tar")
    # for saving
    model_path = os.path.join(args.ckpt_dir, "checkpoint.pth.tar")

    training_state = {'model' : model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                      'epoch': epoch,
                      'rng' : rng}
    if amp is not None:
        training_state.update({'amp': amp.state_dict()})
    
    if args.checkpoint_path: # restricting from saving in local
        # first save to temp file
        torch.save(training_state, temp_path)
        # according to the GNU spec of rename, the state of checkpoint_path
        # is atomic, i.e. it will either be modified or not modified, but not in
        # between, during a system crash (i.e. preemtion)
        os.replace(temp_path, checkpoint_path)
        
    # also save a permanent copy
    torch.save(training_state, model_path)
    msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path + '\n and \n' +  model_path
    logger.add_line(msg)



def create_or_restore_training_state3(args, model, optimizer, logger, amp):
    
    def _system_restore(training_state, model, optimizer, amp):
        model.load_state_dict(training_state['model'])
        optimizer.load_state_dict(training_state['optimizer'])
        start_epoch = training_state['epoch']+1
        rng = training_state['rng'].cpu()
        torch.random.set_rng_state(rng)
        if amp is not None:
                amp.load_state_dict(training_state['amp'])
        
        return model, optimizer, start_epoch, amp, rng
    
    logger.add_line("looking for checkpoint restoration...")    
    path = None
    start_epoch=0
    rng = torch.random.get_rng_state()
    if os.path.isfile(os.path.join(args.ckpt_dir, 'checkpoint.pth.tar')):
        path = os.path.join(args.ckpt_dir, 'checkpoint.pth.tar')
    if args.resume and os.path.isfile(os.path.join(args.resume, 'checkpoint.pth.tar')):
        path = os.path.join(args.resume, 'checkpoint.pth.tar')
        
    if path is not None:
        if args.gpu is None:
            training_state = torch.load(path)
        else:
            # Map model to be loaded to specified single gpu.
            training_state = torch.load(path, map_location='cuda:{}'.format(args.gpu))
        model, optimizer, start_epoch, amp, rng = _system_restore(training_state, 
                                                                           model, optimizer, amp)

        logger.add_line(f"training state restored from {path} at epoch: {start_epoch}")
        
    else:
        logger.add_line(f"No checkpoint found either at resume: {args.resume} or ckpt_dir: {args.ckpt_dir}")

    return model, optimizer, start_epoch, amp, rng


def commit_state3(args, model, optimizer, epoch, amp, rng, logger):
    
    temp_path = os.path.join(args.ckpt_dir, "temp.pth.tar")
    model_path = os.path.join(args.ckpt_dir, "checkpoint.pth.tar")

    training_state = {'model' : model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                      'epoch': epoch,
                      'rng' : rng}
    if amp is not None:
        training_state.update({'amp': amp.state_dict()})
    
    # first save to temp file
    torch.save(training_state, temp_path)
    # according to the GNU spec of rename, the state of checkpoint_path
    # is atomic, i.e. it will either be modified or not modified, but not in
    # between, during a system crash (i.e. preemtion)
    os.replace(temp_path, model_path)
    msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + model_path
    logger.add_line(msg)