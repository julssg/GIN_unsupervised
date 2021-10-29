import os
from os import listdir
from os.path import isfile, join
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import seaborn as sns
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical, CCA
import matplotlib.pyplot as plt
from data import make_dataloader_emnist
import csv
import numpy as np
from metrics import mean_corr_coef


def cca_evaluation(args, GIN, save_dir):

    torch.cuda.memory_summary(device=None, abbreviated=False)
    torch.cuda.empty_cache()

    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f)) and '.pt' in f ]
    num_runs = len(saved_models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if GIN.dataset == 'EMNIST':
        # 1: load test data set
        batch_size = n = 2000
        test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir)
        dims = [100, 200 , 300, 400, 500] # np.arange(10, 350, 10)

        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx < 1:
                data_val = data.to(device)
    
    else:
        data_val = GIN.data.to(device)
        n = args.n_data_points
        dims = [10]

    score_values_list =[]
    mean_cc_list = []
    mean_cc_list_train = []

    mean_cc_std_list = []
    mean_cc_std_list_train = []

    dict = {'dimension':[],
        'mcc_value':[],
        'data':[]
       }
  
    df = pd.DataFrame(dict)

    for dim in dims:
        mean_cc_list_models = []
        mean_cc_list_models_train = []
        for i in range(num_runs):
            for j in range(num_runs):
                if j != i:
                
                    print(f"Using models {saved_models[i]} as reference model and {saved_models[j]} model to compare with.")
                    
                    model = GIN.to(device)
                    data = torch.load(saved_models[i])
                    model.load_state_dict(data['model'])
                    model.to(device)
                    model.eval()

                    z_ref_val = model(data_val)[0].cpu().detach() 

                    del model

                    model = GIN
                    data = torch.load(saved_models[j])
                    model.load_state_dict(data['model'])
                    model.to(device)
                    model.eval()
                    z_comp_val = model(data_val)[0].cpu().detach()

                    del model

                    # mean_cc_list = []

                # for dim in dims:
                    # test different cca evaluation: 
                    # z_ref_val = np.random.rand(batch_size,784)
                    # z_comp_val = np.random.rand(batch_size,784)
                    # print((z_ref_val[:int(batch_size/2)]).shape)

                    # cca_dim = dim
                    # cca = CCA(n_components=cca_dim, max_iter=500)
                    # cca.fit(z_ref_val[:int(batch_size/2)], z_comp_val[:int(batch_size/2)])
                    # res_out = cca.transform(z_ref_val[:int(batch_size/2)], z_comp_val[:int(batch_size/2)])
                    # mcc_weak_out = mean_corr_coef(res_out[0], res_out[1], method = 'spearman')
                    # res_in = cca.transform(z_ref_val[int(batch_size/2):], z_comp_val[int(batch_size/2):])
                    # mcc_weak_in = mean_corr_coef(res_in[0], res_in[1])
                    # print('mcc weak in: ', mcc_weak_in, ' --- ccadim = ', cca_dim)
                    # print('mcc weak out: ', mcc_weak_out, ' --- ccadim = ', cca_dim)

                    # cca = CCA(n_components=cca_dim, max_iter=500)
                    # cca.fit(z_comp_val[:int(batch_size/2)], z_ref_val[:int(batch_size/2)])
                    # res_out = cca.transform(z_comp_val[:int(batch_size/2)], z_ref_val[:int(batch_size/2)])
                    # mcc_weak_out = mean_corr_coef(res_out[0], res_out[1])
                    # res_in = cca.transform(z_comp_val[int(batch_size/2):], z_ref_val[int(batch_size/2):])
                    # mcc_weak_in = mean_corr_coef(res_in[0], res_in[1])
                    # print('mcc weak in: ', mcc_weak_in, ' --- ccadim = ', cca_dim)
                    # print('mcc weak out after swapping: ', mcc_weak_out, ' --- ccadim = ', cca_dim)

                    X = z_ref_val # + np.random.normal(size=784 * n).reshape((n, 784))
                    Y = z_comp_val # z_comp_val # + np.random.normal(size=4 * n).reshape((n, 4))

                    X_train = X[:n // 2]
                    Y_train = Y[:n // 2]
                    X_test = X[n // 2:]
                    Y_test = Y[n // 2:]

                    # print("Corr(X)")
                    # print(np.round(np.corrcoef(X.T), 2))
                    # print("Corr(Y)")
                    # print(np.round(np.corrcoef(Y.T), 2))

                    # #############################################################################
                    # Canonical (symmetric) PLS

                    # Transform data
                    # ~~~~~~~~~~~~~~
                    plsca = PLSCanonical(n_components=dim, max_iter=1500)
                    # plsca = CCA(n_components=dim)
                    plsca.fit(X_train, Y_train)
                    X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
                    X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

                    # Compute Mean correlation coefficent over all components
                    mean_cc = 0 
                    mean_cc_train = 0 
                    for k in range(dim):
                        mean_cc += np.corrcoef(X_test_r[:, k], Y_test_r[:, k])[0, 1]
                        mean_cc_train += np.corrcoef(X_train_r[:, k], Y_train_r[:, k])[0, 1]
                        # with open(f'{save_dir}\cc_per_dim_{dim}_train.csv', 'a') as csvfile:
                        #      filewriter = csv.writer(csvfile, delimiter=',',
                        #                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            
                        #      filewriter.writerow([k, np.corrcoef(X_train_r[:, k], Y_train_r[:, k])[0, 1]])

                    mean_cc /= dim 
                    mean_cc_train /= dim 
                    mean_cc_list_models.append(mean_cc)
                    mean_cc_list_models_train.append(mean_cc_train)

                    df.loc[len(df.index)] = [f'{dim}', mean_cc, 'test'] 
                    df.loc[len(df.index)] = [f'{dim}', mean_cc_train, 'val'] 

        mean_cc = np.mean(np.array(mean_cc_list_models))
        mean_cc_std = np.std(np.array(mean_cc_list_models))

        mean_cc_train = np.mean(np.array(mean_cc_list_models_train))
        mean_cc_train_std = np.std(np.array(mean_cc_list_models_train))

        print(f"The mean correlation coefficient over serveral models\
             using {dim} components on validation data is: {mean_cc_train} +- {mean_cc_train_std}.")
        print(f"The mean correlation coefficient over serveral models\
             using {dim} components on test data is: {mean_cc} +- {mean_cc_std}.")

                    # plsca = PLSCanonical(n_components=dim)
                    # # plsca = CCA(n_components=784)
                    # X_train, Y_train = Y_train, X_train
                    # X_test, Y_test = Y_test, X_test
                    # plsca.fit(X_train, Y_train)
                    # X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
                    # X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

                    # # Compute Mean correlation coefficent over all components
                    # mean_cc = 0 
                    # for k in range(dim):
                    #     mean_cc += np.corrcoef(X_test_r[:, k], Y_test_r[:, k])[0, 1]
                    # mean_cc /= dim 
                    # print(f"The mean correlation coefficient using {dim} components, X and Y swaped, is: {mean_cc}.")
                    
        mean_cc_list.append(mean_cc)
        mean_cc_list_train.append(mean_cc_train)

        mean_cc_std_list.append(mean_cc_std)
        mean_cc_std_list_train.append(mean_cc_train_std)


    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x="dimension", y="mcc_value", hue="data",
                 data=df, palette="Set3")
    plt.savefig(f'{save_dir}\plot_find_good_dim.pdf')
                # print(mean_cc_list)
                # plt.plot(dims, np.array(mean_cc_list))
                # plt.show()
                # plt.savefig(f'{save_dir}\mcc_scores_models_{i}_{j}.pdf')


                    # # reference:
                    # if i == 0:
                    #     try:
                    #         print(f"Fitting reference CCA with {dim} components....")
                    #         cca = CCA(n_components=dim)
                    #         cca.fit(z_ref_val, z_ref_val)
                    #         score_ref_val =  cca.score(z_ref_val , z_ref_val)

                    #         cca = CCA(n_components=dim)
                    #         cca.fit(z_comp_val, z_comp_val)
                    #         score_comp_val =  cca.score(z_comp_val , z_comp_val)
                    #         print(f"The reference scores for model {i} is {score_ref_val} and for {j} is {score_comp_val} ")

                    #     except Exception as e:
                    #         print(e)
                    # else: 
                    #     score_ref_val = "see above"
                    #     score_comp_val = "see above"

                    
                    # try:
                    #     print(f"Fitting CCA with {dim} components....")
                    #     cca = CCA(n_components=dim, max_iter=5000)
                    #     cca.fit(z_ref_val, z_comp_val)
                    #     print(f"Evaluating CCA with {dim} components for latent spaces....")
                    #     score_val = cca.score(z_ref_val , z_comp_val)
                    #     print(f"The validation score for models {i} and {j} for cca in latent space is: {score_val}")
                    #     try:
                    #         norm_score_val = score_val**2/(score_ref_val*score_comp_val)
                    #     except Exception as e:
                    #         print(e)
                    #         norm_score_val = None
                    #     print(f"The validation score including the reference value for models {i} and {j} for cca in latent space is: {norm_score_val}")
                    #     score_values_list.append(score_val)

                    #     with open(f'{save_dir}\cca_score_{dim}.csv', 'a') as csvfile:
                    #         filewriter = csv.writer(csvfile, delimiter=',',
                    #                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            
                    #         filewriter.writerow([saved_models[i], saved_models[j], score_ref_val, score_comp_val, score_val, norm_score_val ])

                    #     del cca
                        
                    # except Exception as e:
                    #     print(e)
                    #     print("Continue with next model comparison.")
        
        # mean_score_value = sum(score_values_list)/len(score_values_list)

        # with open(f'{save_dir}\cca_score_{dim}.csv', 'a') as csvfile:
        #     filewriter = csv.writer(csvfile, delimiter=',',
        #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     filewriter.writerow(['mean value',' ' , ' ' , ' ' , mean_score_value, ' ' ])


def load(model_init, fname, device):
    model = model_init
    data = torch.load(fname)
    model.load_state_dict(data['model'])
    model.eval()
    model.to(device)
    return model
    
# def learn_permutation(y_ref, y_comp, n_clusters):
#     permutation = []
#     for cluster in range(n_clusters):
#         idx_of_samle_classes = [i for i, j in enumerate(y_ref) if j == cluster ]
#         # print(idx_of_samle_classes)
#         # print(len(idx_of_samle_classes))
#         if idx_of_samle_classes:
#             y_permuted_idx = mode(y_comp[idx_of_samle_classes])
#         else:
#             y_permuted_idx = None
#         print("Pertupation pair: ", cluster, y_permuted_idx)

#         permutation.append(y_permuted_idx)

# def permutate(y):
#     permutation = np.random.permutation(40)
#     y_permuted = np.array([permutation[k] for k in y])
#     print("Permutation matrix", permutation)
#     return y_permuted
