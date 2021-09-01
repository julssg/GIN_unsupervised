import argparse
import os
from os import listdir
from os.path import isfile, join
import torch
from time import time
from model import GIN

parser = argparse.ArgumentParser(description='Experiments on EMNIST with GIN (training script)')
parser.add_argument('--save_dir', type=str, default=None,
                    help='Folder where different train runs are saved.')

args = parser.parse_args()

assert os.path.isdir(args.save_dir) == True, 'Path does not exists'

def mcc_evaluation(model, save_dir):
    saved_models = [f for f in listdir(save_dir) if isfile(join(save_dir, f))]
    num_runs = len(saved_models)

    for i in range(num_runs):
        for j in range(num_runs):
            if j != i:
                model_ref = torch.load(model, saved_models[i])
                model_comp = torch.load(model, saved_models[j])
