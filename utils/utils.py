import os
import torch
import random
import logging
import numpy as np
import datetime
import json


def mkdirs(dirpath):
    """
    Create directory if it doesn't exist
    
    Args:
        dirpath: Path of the directory to be created
    """
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def fix_random_seed(seed):
    """
    Fix all random seeds
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, args, level=logging.INFO):
        
        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print(start_time)

        if args.partition == "drichlet":
            result_logdir = os.path.join(args.logdir, args.algorithm, args.dataset, args.partition, str(args.drichlet_beta), args.model)
        elif args.partition == "pathological":
            result_logdir = os.path.join(args.logdir, args.algorithm, args.dataset, args.partition, str(args.shard_per_user), args.model)
        
        if not os.path.exists(result_logdir):
            os.makedirs(result_logdir)

        log_path = 'logs-%d-%s' % (args.seed, start_time) + '.log'

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            filename=os.path.join(result_logdir, log_path),
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            level=level,
            filemode='w'
        )
        
        self.logger = logging.getLogger()
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)