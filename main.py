import numpy as np
import os
import sys
import argparse
import copy
from hdf5storage import loadmat
from hdf5storage import savemat

sys.path.append('model')
sys.path.append('data')

from SGPRN import SGPRN
from DataProcess import process


def run(args):
    
    domain = args.domain
    kernel = args.kernel
    device = args.device
    rank = int(args.rank)
    maxIter = int(args.maxIter)
    interval = int(args.interval)

    print('Experiment summary: ')
    print(' - Domain name:', domain)
    print(' - Device id:', device)
    print(' - Cov Func:', kernel)
    print(' - rank:', rank)
    print(' - maxIter:', maxIter)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    print('Using GPU', device)
    
    res_path = 'results'
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
            
    trial = 1
    data = process(domain)

    signature = domain + '_rank' + str(rank) + '_t' + str(trial)
    cfg={
        'data': data,
        'signature': signature,
        'jitter': 1e-3,
        'init_std': 1,
        'epochs': maxIter,
        'interval':interval,
        'alpha':1e-3,
        'ns': 100,
        'Q': rank,
        'kernel': kernel,
    }
   
    try:
        model = SGPRN(cfg, label=signature, init_type=2)
        res = model.fit()
        cfg['result'] = res
        res_save_path = os.path.join(res_path, signature)
        savemat(res_save_path, cfg, format='7.3')
        print('results saved to', res_save_path + '.mat')
    except:
        print('Exceptions occurs during training...')

    

if __name__== "__main__" :
    
    args = argparse.ArgumentParser()
    args.add_argument("--domain", "-d", dest="domain", type=str, required=True)
    args.add_argument("--kernel", "-k", dest="kernel", type=str, required=True)
    args.add_argument("--device", "-v", dest="device", type=str, required=True)
    args.add_argument("--rank", "-r", dest="rank", type=str, required=True)
    args.add_argument("--maxIter", "-t", dest="maxIter", type=str, required=True)
    args.add_argument("--interval", "-i", dest="interval", type=str, required=True)
    args = args.parse_args()
    
    run(args)