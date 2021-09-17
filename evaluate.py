import argparse
import os
from os import listdir
from os.path import isfile, join
import torch
import statistics
from statistics import mode
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

import numpy as np


from data import make_dataloader_emnist

def cca_evaluation(args, GIN, save_dir):

    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f))]
    num_runs = len(saved_models)

    # 1: load test data set
    batch_size = 20000
    test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir)
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx < 1:
            data_val = data[:int(batch_size/10)].to(device)
            data_test = data[int(batch_size/10)+1:].to(device)

    for i in range(num_runs):
        for j in range(num_runs):
            if j != i:
                print(f"Using models {saved_models[i]} as reference model and {saved_models[j]} model to compare with.")
                
                model = GIN
                data = torch.load(saved_models[i])
                model.load_state_dict(data['model'])
                model.to(device)
                model.eval()

                # model_ref = load(GIN, saved_models[i], device) 
                # model_ref.eval()
                z, logdet_J = model.net(data_val)
                z_ref_val = z.detach()
                y_ref_val = model.predict_y(z_ref_val.to(device)).reshape(-1, 1)
                #categ = np.arange(40).reshape(-1, 1)
                #print(categ)
                y_ref_val_encoded = OneHotEncoder(categories=[range(40)]*1).fit_transform(y_ref_val).toarray()
                # do one enclode but with all 40 classes ( if eg label 30 not in val data, then not in consideration)
                # y_ref_val_permutated = permutate(y_ref_val)

                # # np.set_printoptions(precision=None, threshold=20000, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None)
                # conf = confusion_matrix(y_ref_val, y_ref_val_permutated)
                # # print(conf)
                # learned_permutation = np.argmax(conf, axis=1)
                # print("learned permutation", learned_permutation)

                # z, logdet_J = model.net(data_test)
                # z_ref_test = z.detach()
                # y_ref_test = model.predict_y(z_ref_test.to(device)).reshape(-1, 1)

                model = GIN
                data = torch.load(saved_models[j])
                model.load_state_dict(data['model'])
                model.to(device)
                model.eval()
                # model_comp =  load(GIN, saved_models[j], device)
                # model_comp.eval()
                z, logdet_J = model.net(data_val)
                z_comp_val = z.detach()
                y_comp_val = model.predict_y(z_comp_val.to(device)) # .reshape(-1, 1)

                # z, logdet_J = model.net(data_test)
                # z_comp_test = z.detach()
                # y_comp_test = model.predict_y(z_comp_test.to(device)).reshape(-1, 1)
                np.set_printoptions(precision=None, threshold=20000, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None)
                conf = confusion_matrix(y_ref_val, y_comp_val)

                # if it would be identifiable, in each row of confusion matrix at least 1 non zero element ( all zero if not sample of this class )
                # but how to measure 'degree' of non-identifiablility 

                # print(conf)
                # learned_permutation = np.argmax(conf, axis=1)
                # print("learned permutation", learned_permutation)

                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(y_ref_val_encoded, y_comp_val)
                probs = clf.predict_proba(y_ref_val_encoded[:5])
                print(y_ref_val[:5], y_comp_val[:5])
                print(probs)
                exit(1)

                # cca = CCA(n_components=1)
                # cca.fit(y_ref_val, y_ref_val_permutated)

                # print("y_ref_val, y_ref_val_permutated", y_ref_val.reshape(1,-1), y_ref_val_permutated.reshape(1,-1))

                # y_ref_val_permutated_predict = cca.transform(y_ref_val)
                # print("y_ref_val, y_ref_val_permutated", y_ref_val_permutated_predict.reshape(1,-1), y_ref_val_permutated.reshape(1,-1))
                # print(y_ref_val_permutated_predict.reshape(1,-1) - y_ref_val_permutated.reshape(1,-1))

                # y_ref_val_permutated_predict = cca.inverse_transform(y_ref_val)
                # print("y_ref_val, y_ref_val_permutated", y_ref_val_permutated_predict.reshape(1,-1), y_ref_val_permutated.reshape(1,-1))
                # print(y_ref_val_permutated_predict.reshape(1,-1) - y_ref_val_permutated.reshape(1,-1))

                # y_ref_val_t, y_comp_val_t = cca.transform(y_ref_val, y_comp_val)
                # y_ref_test_t, y_comp_test_t = cca.transform(y_ref_test, y_comp_test)
                # score_val = cca.score(y_ref_val, y_ref_val_permutated)
                # score_test = cca.score(y_ref_test, y_ref_test )

                # print(f"The validation score for models {i} and {j} after linear transformation is: {score_val}")
                # print(f"The test score for models {i} and {j} after linear transformation is: {score_test}")
    

def mcc_evaluation(args, GIN, save_dir):
    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f))]
    num_runs = len(saved_models)

    # 1: load test data set
    batch_size = 20000
    test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir)
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx < 1:
            data_val = data[:int(batch_size/10)].to(device)
            data_test = data[int(batch_size/10)+1:].to(device)

    for i in range(num_runs):
        for j in range(num_runs):
            if j != i:
                print(f"Using models {saved_models[i]} as reference model and {saved_models[j]} model to compare with.")
                
                model = GIN
                data = torch.load(saved_models[i])
                model.load_state_dict(data['model'])
                model.to(device)
                model.eval()

                # model_ref = load(GIN, saved_models[i], device) 
                # model_ref.eval()
                z, logdet_J = model.net(data_val)
                z = z.detach()
                y_ref = model.predict_y(z.to(device))


                model = GIN
                data = torch.load(saved_models[j])
                model.load_state_dict(data['model'])
                model.to(device)
                model.eval()
                # model_comp =  load(GIN, saved_models[j], device)
                # model_comp.eval()
                z, logdet_J = model.net(data_val)
                z = z.detach()
                y_comp = model.predict_y(z.to(device))

                # # 3: learn linear permutation between classes between both models 
                # # print(len(y_ref), len(y_comp))
                # print(y_comp)

                # print(y_ref - y_comp)
                # print(sum(y_ref - y_comp))
                learned_permutation = learn_permutation(y_ref, y_comp, args.n_clusters)
                print(learned_permutation )

                exit(1)


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
    permutation = []
    for cluster in range(n_clusters):
        idx_of_samle_classes = [i for i, j in enumerate(y_ref) if j == cluster ]
        # print(idx_of_samle_classes)
        # print(len(idx_of_samle_classes))
        if idx_of_samle_classes:
            y_permuted_idx = mode(y_comp[idx_of_samle_classes])
        else:
            y_permuted_idx = None
        print("Pertupation pair: ", cluster, y_permuted_idx)

        permutation.append(y_permuted_idx)

def permutate(y):
    permutation = np.random.permutation(40)
    y_permuted = np.array([permutation[k] for k in y])
    print("Permutation matrix", permutation)
    return y_permuted


