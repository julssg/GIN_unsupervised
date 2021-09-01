import argparse
import os
from os import listdir
from os.path import isfile, join
import torch

from data import make_dataloader_emnist


def mcc_evaluation(model, save_dir, data_root_dir):
    saved_models = [f for f in listdir(save_dir) if isfile(join(save_dir, f))]
    num_runs = len(saved_models)

    # 1: load test data set

    test_loader  = make_dataloader_emnist(batch_size=1000, train=False, root_dir=data_root_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(num_runs):
        for j in range(num_runs):
            if j != i:
                model_ref = torch.load(model, saved_models[i])
                model_comp = torch.load(model, saved_models[j])

                model_ref.eval().to(device)
                model_comp.eval().to(device)

                for batch_idx, (data, target) in enumerate(test_loader):
                # 2: classify 30% data with model_ref and model_comp   # only use 10000 data
                    data.to(device)
                    if batch_idx < 3:
                        z, logdet_J = model_ref.net(data)
                        y_ref = model_ref.predict_y(z)

                        z, logdet_J = model_comp.net(data)
                        y_comp = model_comp.predict_y(z)
                        # 3: learn linear permutation between classes between both models 
                            # Code here

                    if batch_idx < 11:
                        z, logdet_J = model_ref.net(data)
                        y_ref = model_ref.predict_y(z)

                        z, logdet_J = model_comp.net(data)
                        y_comp = model_comp.predict_y(z)
                # 4: classify rest data with both models and use classification model_ref = y_true and model_comp = y_pred
                # 5: calculate MCC
                    # Code here

