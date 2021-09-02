import argparse
import os
from os import listdir
from os.path import isfile, join
import torch
import statistics
from statistics import mode

from data import make_dataloader_emnist


def mcc_evaluation(args, model, save_dir):
    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f))]
    num_runs = len(saved_models)

    # 1: load test data set

    test_loader  = make_dataloader_emnist(batch_size=10000, train=False, root_dir=args.data_root_dir)
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(num_runs):
        for j in range(num_runs):
            if j != i:
                print(f"Using models {saved_models[i]} as reference model and {saved_models[j]} model to compare with.")

                # model_comp =  load(model, saved_models[4], device)

                for batch_idx, (data, target) in enumerate(test_loader):
                # 2: classify 30% data with model_ref and model_comp   # only use 10000 data
                    data.to(device)
                    if batch_idx < 1:

                        model_ref = load(model, saved_models[i], device) 
                        z, logdet_J = model_ref.net(data)
                        y_ref = model_ref.predict_y(z.to(device))
                        print(y_ref)

                        model_comp =  load(model, saved_models[j], device)
                        z, logdet_J = model_comp.net(data)
                        y_comp = model_comp.predict_y(z.to(device))
                        # 3: learn linear permutation between classes between both models 
                        # print(len(y_ref), len(y_comp))
                        print(y_comp)

                        print(y_ref - y_comp)
                        print(sum(y_ref - y_comp))
                        learn_permutation(y_ref, y_comp, args.n_clusters)


                    # elif batch_idx < 11:
                       # z, logdet_J = model_ref.net(data)
                       # y_ref = model_ref.predict_y(z.to(device))

                        # z, logdet_J = model_comp.net(data)
                        # y_comp = model_comp.predict_y(z.to(device))
                # 4: classify rest data with both models and use classification model_ref = y_true and model_comp = y_pred
                # 5: calculate MCC
                    # Code here

def load(model_init, fname, device):
    model = model_init
    data = torch.load(fname)
    model.load_state_dict(data['model'])
    model.eval()
    model.to(device)
    return model
    
def learn_permutation(y_ref, y_comp, n_clusters):

    for cluster in range(n_clusters):
        idx_of_samle_classes = [i for i, j in enumerate(y_ref) if j == cluster ]
        # print(idx_of_samle_classes)
        # print(len(idx_of_samle_classes))
        if idx_of_samle_classes:
            y_permuted = mode(y_comp[idx_of_samle_classes])
        else:
            y_permuted = None
        print("Pertupation pair: ", cluster, y_permuted)


