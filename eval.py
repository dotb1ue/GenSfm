import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import random

from trellis import datasets, evaluators
from trellis.utils.dist_utils import setup_dist, destroy_dist
from trellis.utils.general_utils import apply_config_overrides, setup_rng


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
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    
    # Log rank information
    print(f'Process started - Node: {cfg.node_rank}, Local Rank: {local_rank}, Global Rank: {rank}, World Size: {world_size}')
    
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Load data
    if hasattr(cfg, 'eval_dataset'):
        try:
            dataset = getattr(datasets, cfg.eval_dataset.name)(cfg.data_dir, is_eval=True, **cfg.eval_dataset.args)
        except:
            dataset = getattr(datasets, cfg.eval_dataset.name)(cfg.data_dir, **cfg.eval_dataset.args)
    else:
        try:
            dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, is_eval=True, **cfg.dataset.args)
        except:
            dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)

    # Build evaluator
    model_cfgs = cfg.models if hasattr(cfg, 'models') else {}
    eval_model_cfgs = cfg.eval_models if hasattr(cfg, 'eval_models') else {}
    model_cfgs.update(eval_model_cfgs)
    evaluator = getattr(evaluators, cfg.evaluator.name)(model_cfgs, dataset, **cfg.evaluator.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, load_step=cfg.load_ckpt)

    # Run evaluation
    evaluator.run()

    # Clean up
    if world_size > 1:
        destroy_dist()


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, default='', help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default=None, help='Port for distributed training')
    ## Evaluation
    parser.add_argument('--max_num_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    ## config overrides
    parser.add_argument('--override', nargs='*', default=[], help='Config overrides in dot notation, e.g. evaluator.args.batch_size_per_gpu=4')
    opt = parser.parse_args()
    opt.config = opt.config if opt.config != '' else os.path.join(opt.output_dir, 'config.json')
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
    if opt.max_num_samples is not None:
        cfg.evaluator.args['max_num_samples'] = opt.max_num_samples
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'eval_command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'eval_config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Run
    cfg = find_ckpt(cfg)
    if cfg.num_gpus > 1:
        mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    else:
        main(0, cfg)
