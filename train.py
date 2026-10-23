import os
import sys
import json
import glob
import argparse
import logging
import builtins
from datetime import datetime
from easydict import EasyDict as edict

import torch
import torch._dynamo
import torch.multiprocessing as mp
import random

# Raise dynamo's per-function cache size so compiled sparse ops that see a small
# but variable set of input shapes/dtypes (SparseLinear, SparseTensor.replace,
# attention blocks whose _layout classes vary with batch grouping) can keep all
# observed specializations cached instead of falling back to eager once the
# default limit (8) is exceeded.
torch._dynamo.config.cache_size_limit = 64

from trellis import datasets, trainers
from trellis.utils.dist_utils import setup_dist, destroy_dist
from trellis.utils.general_utils import apply_config_overrides, setup_rng


class FileOnlyHandler(logging.Handler):
    """
    A handler that writes only to a file, not to console.
    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.file_handler = logging.FileHandler(filename)
        
    def emit(self, record):
        # Write only to file
        self.file_handler.emit(record)
        
    def close(self):
        self.file_handler.close()
        super().close()


class StderrToLogger:
        def __init__(self, logger):
            self.logger = logger
            self.stderr = sys.stderr
            
        def write(self, message):
            if message.strip():
                self.logger.error(message.strip())
            self.stderr.write(message)
                
        def flush(self):
            self.stderr.flush()


class PrintToLogger:
    """
    A class that redirects print() calls to logging.
    """
    def __init__(self, logger):
        self.logger = logger
        self.original_print = print
        
    def __call__(self, *args, **kwargs):
        # Convert print arguments to a string
        message = ' '.join(str(arg) for arg in args)
        # Log the message
        self.logger.info(message)
        # Also call the original print for console output
        self.original_print(*args, **kwargs)


def setup_logging(output_dir, node_rank=0, local_rank=0, world_size=1):
    """
    Setup logging to both console and file.
    All processes will create log files with rank-specific names.
    """
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log filename with timestamp and rank info
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if world_size > 1:
        # For multi-GPU, include rank info
        rank = node_rank * world_size + local_rank
        log_file = os.path.join(log_dir, f'training_{timestamp}_rank_{rank:02d}.log')
    else:
        # For single GPU, just use timestamp
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging with custom handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our custom handler (file only, console output handled by PrintToLogger)
    handler = FileOnlyHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Redirect stderr to logging
    sys.stderr = StderrToLogger(logger)
    
    # Monkey-patch print function to use logging
    builtins.print = PrintToLogger(logger)
    
    return log_file


def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def main(local_rank, cfg):
    # Enable TF32
    torch.set_float32_matmul_precision('high')

    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    
    # Setup logging first (before distributed setup to capture any errors)
    log_file = setup_logging(cfg.output_dir, cfg.node_rank, local_rank, world_size)
    
    # Log rank information
    print(f'Process started - Node: {cfg.node_rank}, Local Rank: {local_rank}, Global Rank: {rank}, World Size: {world_size}, Logging to: {log_file}')
    
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Load data
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(cfg.models, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()

    # Clean up
    if world_size > 1:
        destroy_dist()


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--auto_retry', type=int, default=3, help='Number of retries on error')
    ## debug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default=None, help='Port for distributed training')
    ## config overrides
    parser.add_argument('--override', nargs='*', default=[], help='Config overrides in dot notation, e.g. models.denoiser.args.num_view_embeds=4')
    opt = parser.parse_args()
    if opt.auto_retry < 0:
        parser.error('--auto_retry must be non-negative')

    # overwite
    opt.node_rank = int(os.environ.get('SLURM_PROCID', opt.node_rank))
    print(f"node rank: {opt.node_rank}")


    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    opt.master_port = str(random.randint(10000, 20000)) if opt.master_port is None else opt.master_port
    ## Load config
    config = json.load(open(opt.config, 'r'))
    ## Apply config overrides
    if opt.override:
        config = apply_config_overrides(config, opt.override)
        print('\n\nConfig overrides:')
        for o in opt.override:
            print(f'  {o}')
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f'Error: {e}')
                if rty + 1 == cfg.auto_retry:
                    raise
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')
