import os
from os import listdir
from os.path import isfile, join
from numpy.core.numeric import cross
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import seaborn as sns
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data import make_dataloader_emnist
import csv
import numpy as np
from metrics import mean_corr_coef


def cca_evaluation(args, GIN, save_dir, n_data_points=None ):

    print("############## Cross Model Identifiability Evaluation ################")

    torch.cuda.memory_summary(device=None, abbreviated=False)
    torch.cuda.empty_cache()

    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f)) and '.pt' in f ]
    num_runs = len(saved_models)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    methods = [ "PLSCan", "CCA", "PCAfull+PLSCan"] # caution bug here: PCAfull transforms data!
    method = methods[0]

    if GIN.dataset == 'EMNIST':
        # 1: load test data set
        batch_size = n = 5000
        test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir, shuffle=False)
        dims = [20] # np.arange(5, 100, 5) #  [40, 50, 100,200,300,400,450] 

        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx < 1:
                data_val = data.to(device)
    
    else:
        n = n_data_points # args.n_data_points
        val_batches = 2
        num_batches = 3
        batch_size = int(n/num_batches)
        data_val = GIN.data.to(device)
        n = data_val.shape[0]
        test_loader = None 
        # dims = [10]
        dims = [5, 10] # np.arange(1 , GIN.n_dims+1 , 1)
        print(f"The number of used clusters is: {GIN.n_classes} ")


    dict = {'dimension':[],
        'mcc_value':[],
        'data':[],
        'n_data_points':[],
        'method':[]
       }
  
    df = pd.DataFrame(dict)
    for method in methods:
        for dim in dims:
            mean_cc_list_models = []
            mean_cc_list_models_train = []
            for i in range(num_runs):
                for j in range(num_runs):
                    if (j != i) and (j>i) :
                    
                        print(f"Using models {saved_models[i]} as reference model and {saved_models[j]} model to compare with.")
                
                        z_ref_val, z_ref_test = get_latent_space_batches(GIN, args, \
                            saved_models[i], test_loader, batch=False, batch_size=batch_size, num_batches=num_batches, val_batches=val_batches)
                        z_comp_val, z_comp_test = get_latent_space_batches(GIN, args, \
                            saved_models[j], test_loader, batch=False, batch_size=batch_size, num_batches=num_batches, val_batches=val_batches)

                        print("the shape of latent space data is", z_ref_val.shape, z_ref_test.shape)
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

                        # X = z_ref_val # + np.random.normal(size=784 * n).reshape((n, 784))
                        # Y = z_comp_val # z_comp_val # + np.random.normal(size=4 * n).reshape((n, 4))

                        #################### Find good dim PCA ######################

                        # pca = PCA().fit(z_ref_val)

                        # plt.rcParams["figure.figsize"] = (12,6)

                        # fig, ax = plt.subplots()
                        # xi = np.arange(1, 785, step=1)
                        # y = np.cumsum(pca.explained_variance_ratio_)

                        # plt.ylim(0.0,1.1)
                        # plt.plot(xi, y, marker='o', linestyle='--', color='b')

                        # plt.xlabel('Number of Components')
                        # plt.xticks(np.arange(0, 785, step=50)) #change from 0-based array index to 1-based human-readable label
                        # plt.ylabel('Cumulative variance (%)')
                        # plt.title('The number of components needed to explain variance')

                        # plt.axhline(y=0.95, color='r', linestyle='-')
                        # plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

                        # ax.grid(axis='x')
                        # # plt.show()
                        # plt.savefig(f'{save_dir}\PCA_find_n.pdf')

                        #################### Apply PCA ##################
                        if "PCA" in method: 
                            if GIN.dataset == 'EMNIST':
                                PCA_dim = 450 

                                pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                                z_ref_val = pca_ref.transform(z_ref_val)
                                z_ref_test = pca_ref.transform(z_ref_test)

                                pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                                z_comp_val = pca_comp.transform(z_comp_val)
                                z_comp_test = pca_comp.transform(z_comp_test)

                                # print(z_ref_val.shape)

                            else:
                                PCA_dim = GIN.n_dims
                                pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                                z_ref_val = pca_ref.transform(z_ref_val)
                                z_ref_test = pca_ref.transform(z_ref_test)

                                pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                                z_comp_val = pca_comp.transform(z_comp_val)
                                z_comp_test = pca_comp.transform(z_comp_test)

                        ################### Apply Identifyablity metric #

                        X_train = z_ref_val
                        Y_train = z_comp_val
                        X_test = z_ref_test
                        Y_test = z_comp_test

                        # print("Corr(X)")
                        # print(np.round(np.corrcoef(X.T), 2))
                        # print("Corr(Y)")
                        # print(np.round(np.corrcoef(Y.T), 2))

                        # #############################################################################
                        # Canonical (symmetric) PLS

                        # Transform data
                        # ~~~~~~~~~~~~~~
                        if "PLSCan" in method:
                            try:
                                plsca = PLSCanonical(n_components=dim, max_iter=2500)
                                plsca.fit(X_train, Y_train)
                                X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
                                X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

                                # Compute Mean correlation coefficent over all components
                                mean_cc = 0 
                                mean_cc_train = 0 
                                for k in range(dim):
                                    mean_cc += np.corrcoef(X_test_r[:, k], Y_test_r[:, k])[0, 1]
                                    mean_cc_train += np.corrcoef(X_train_r[:, k], Y_train_r[:, k])[0, 1]
    
                                mean_cc /= dim 
                                mean_cc_train /= dim 
                                mean_cc_list_models.append(mean_cc)
                                mean_cc_list_models_train.append(mean_cc_train)

                                df.loc[len(df.index)] = [f'{dim}', mean_cc, 'test', f'{int(n_data_points/1000)} k', method] 
                                df.loc[len(df.index)] = [f'{dim}', mean_cc_train, 'val', f'{int(n_data_points/1000)} k', method] 
                            except Exception as e:
                                print(e)
                        
                        elif "CCA" in method:
                            try:
                                plsca = CCA(n_components=dim, max_iter=2500)
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

                                df.loc[len(df.index)] = [f'{dim}', mean_cc, 'test', f'{int(n_data_points/1000)} k', method] 
                                df.loc[len(df.index)] = [f'{dim}', mean_cc_train, 'val',  f'{int(n_data_points/1000)} k', method] 
                            except Exception as e:
                                print(e)
                        
                        else:
                            print("ERROR: please provide a feature reduction method.")
                            exit(1)
                        
                        try:
                            torch.cuda.empty_cache()
                        except Exception as e:
                            print(e) 

                        print(f"The test MCC value is: {mean_cc}")

            mean_cc = np.mean(np.array(mean_cc_list_models))
            mean_cc_std = np.std(np.array(mean_cc_list_models))

            mean_cc_train = np.mean(np.array(mean_cc_list_models_train))
            mean_cc_train_std = np.std(np.array(mean_cc_list_models_train))

            print(f"The mean correlation coefficient over serveral models\
                using {dim} components an method {method} on validation data is: {mean_cc_train} +- {mean_cc_train_std}.")
            print(f"The mean correlation coefficient over serveral models\
                using {dim} components an method {method}  on test data is: {mean_cc} +- {mean_cc_std}.")

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

    sns.set_theme(style="whitegrid")

    if "PCA" in method:
        ax = sns.boxplot(x="dimension", y="mcc_value", hue="data",
                    data=df, palette="Set3", dodge=False).set(
                        title=f'MCC - GLOW - Feature reduction method: {method} - PCA dim = {PCA_dim} - final dim = {dim} - Dataset {GIN.dataset} - trained with number of clusters = {GIN.n_classes}') #, dodge =False
    else:
        ax = sns.boxplot(x="dimension", y="mcc_value", hue="data",
                    data=df, palette="Set3", dodge=False).set(
                        title=f'MCC - GLOW - Feature reduction method: {method} - final dim = {dim} - Dataset {GIN.dataset} - trained with number of clusters = {GIN.n_classes}') #, dodge =False


    
    if GIN.dataset == 'EMNIST':
        try:
            os.makedirs(f'{save_dir}\produce_experiments')
        except Exception as e:
            print(e) 
        plt.savefig(f'{save_dir}\produce_experiments\informative_name.pdf')


    else:
        try:
            os.makedirs(f'{save_dir}\one_vs_many')
        except Exception as e:
            print(e) 
        plt.savefig(f'{save_dir}\one_vs_many\MCC_values_{GIN.n_classes}_cluster_{n}.pdf')
    
    return df
 

    #plt.savefig(f'{save_dir}\produce_experiments\informative_experiments_our_metric_featuredim20.pdf')
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


def evaluate_stability(args, GIN, save_dir, cross_validation=False):

    print("############## Stability Evaluation ################")

    number_stab_runs = 6
    if GIN.dataset == 'EMNIST':
        dims = [25] # np.arange(5, 110, 10) #  [40, 50, 100,200,300,400,450] 
        batch_size = n = 5000
        test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir, shuffle=False)
        methods = ["PCA+PLSCan", "PCAfull+PLSCan", "PLSCan", "PCA+CCA", "CCA" ]

    else:
        # dims = [GIN.n_dims] 
        test_loader = None
        dims = np.arange(1 , GIN.n_dims+1 , 1)
        batch_size = n = 5000
        methods = ["PCAfull+PLSCan",  "PCAfull+CCA", "CCA" , "PLSCan+center", "PLSCan"]   # "PLSCan",

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f)) and '.pt' in f ]

    print(f"Using models {saved_models[0]} as reference model and {saved_models[1]} model to compare with.")


    if cross_validation:
    #    ' z_ref_set = get_latent_space_batches(GIN, args, \
    #         saved_models[0], batch=cross_validation)
    #     z_comp_set = get_latent_space_batches(GIN, args, \
    #         saved_models[1], batch=cross_validation)'
        
        z_ref_set = get_latent_space_batches(GIN, args, \
            saved_models[0], test_loader, batch=True, batch_size=batch_size, num_batches=5, val_batches=4)
        z_comp_set = get_latent_space_batches(GIN, args, \
            saved_models[1], test_loader, batch=True, batch_size=batch_size, num_batches=5, val_batches=4)


        for method in methods:

            mcc_dims_list = [] 
            mcc_std_dims_list = []

            mcc_dims_list_val = [] 
            mcc_std_dims_list_val = [] 

            for dim in dims:

                mcc_stab_list = [] 
                mcc_stab_list_val = [] 

                for k in range(len(z_ref_set)):

                    z_ref_test = z_ref_set[k]
                    z_comp_test = z_comp_set[k]
                    z_ref_val = np.concatenate([z_ref_set[m] for m in range(len(z_ref_set)) if m != k], axis=0)
                    z_comp_val = np.concatenate([z_comp_set[m] for m in range(len(z_ref_set)) if m != k], axis=0)

                    print(z_comp_val.shape)

                    print(method)

                    #################### Apply PCA ##################
                    if "center" in method:
                        z_ref_val = z_ref_val - z_ref_val.mean(axis=0)
                        z_ref_test = z_ref_test - z_ref_test.mean(axis=0)
                        z_comp_val =  z_comp_val -  z_comp_val.mean(axis=0)
                        z_comp_test = z_comp_test - z_comp_test.mean(axis=0)

                    if "PCA" in method and not "full" in method: 
                        print("############## Apply PCA ################")
                        if GIN.dataset == 'EMNIST':
                            PCA_dim = 450 

                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)

                            # print(z_ref_val.shape)

                        else:
                            PCA_dim = GIN.n_dims
                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)

                    if "PCAfull" in method: 
                        print("############## Apply full PCA ################")
                        if GIN.dataset == 'EMNIST':
                            PCA_dim = 784

                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)

                            # print(z_ref_val.shape)

                        else:
                            PCA_dim = GIN.n_dims
                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)


                    ################### Apply Identifyablity metric #

                    X_val = z_ref_val
                    Y_val = z_comp_val
                    X_test = z_ref_test
                    Y_test = z_comp_test

                    # print("Corr(X)")
                    # print(np.round(np.corrcoef(X.T), 2))
                    # print("Corr(Y)")
                    # print(np.round(np.corrcoef(Y.T), 2))

                    # #############################################################################
                    # Canonical (symmetric) PLS

                    # Transform data
                    # ~~~~~~~~~~~~~~
                    try:
                        if "PLSCan" in method:
                            print("############## Apply PLSCanonical ################")
                            plsca = PLSCanonical(n_components=dim, max_iter=2500)
                        elif "CCA" in method:
                            print("############## Apply CCA ################")
                            plsca = CCA(n_components=dim, max_iter=2500)
                        else:
                            print("ERROR: please provide a feature reduction method.")
                            exit(1)
    
                        plsca.fit(X_val, Y_val)
                        X_train_r, Y_train_r = plsca.transform(X_val, Y_val)
                        X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

                        # Compute Mean correlation coefficent over all components
                        mean_cc = 0 
                        mean_cc_val = 0 
                        for k in range(dim):
                            mean_cc += np.corrcoef(X_test_r[:, k], Y_test_r[:, k])[0, 1]
                            mean_cc_val += np.corrcoef(X_train_r[:, k], Y_train_r[:, k])[0, 1]

                        mean_cc /= dim 
                        mean_cc_val /= dim 
                        mcc_stab_list.append(mean_cc)
                        mcc_stab_list_val.append(mean_cc_val)

                    except Exception as e:
                        print(e)
                    
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(e) 

                    print(f"The test MCC value is: {mean_cc}")
            
                mcc_stab = np.mean(np.array(mcc_stab_list))


                mcc_stab  = np.mean(np.array(mcc_stab_list))
                mcc_stab_std = np.std(np.array(mcc_stab_list))

                mcc_stab_val = np.mean(np.array(mcc_stab_list_val))
                mcc_stab_val_std = np.std(np.array(mcc_stab_list_val))

                print(f"The mean correlation coefficient over serveral runs\
                        using {dim} components on validation data is: {mcc_stab_val} +- {mcc_stab_val_std}.")
                print(f"The mean correlation coefficient over serveral models\
                        using {dim} components on test data is: {mcc_stab} +- {mcc_stab_std }.")

                mcc_dims_list.append(mcc_stab) 
                mcc_std_dims_list.append(mcc_stab_std)
                
                mcc_dims_list_val.append(mcc_stab_val)
                mcc_std_dims_list_val.append(mcc_stab_val_std)
            
            mcc_dims_list = np.array(mcc_dims_list)
            mcc_std_dims_list = np.array(mcc_std_dims_list)

            mcc_dims_list_val = np.array(mcc_dims_list_val)
            mcc_std_dims_list_val = np.array(mcc_std_dims_list_val)

            plt.plot(dims, mcc_dims_list, label=f'{method}_test')
            plt.xlabel("final dimension")
            plt.ylabel("MCC")
            plt.fill_between(dims, mcc_dims_list - mcc_std_dims_list, mcc_dims_list + mcc_std_dims_list, alpha=0.2)
            # plt.plot(dims, mcc_dims_list_val, label='val')
            # plt.fill_between(dims, mcc_dims_list_val - mcc_std_dims_list_val, mcc_dims_list_val + mcc_std_dims_list_val, color='r', alpha=0.2)
            if "PCA" in method:
                plt.title(f'MCC stability - GLOW - Feature reduction method: {method} - PCA dim = {PCA_dim} - \n\
                    final dim = {dim} - Dataset {GIN.dataset} - n_data {args.n_data_points} - n of val data {z_comp_val.shape[0]}, test data {z_comp_test.shape[0]} -\n\
                        trained with number of clusters = {GIN.n_classes} -with cross validation = {cross_validation}', fontsize = 8)

            else:
                plt.title(f'MCC stability - GLOW - Feature reduction method: {method} - \n\
                    final dim = {dim} - Dataset {GIN.dataset} - n_data {args.n_data_points} - n of val data {z_comp_val.shape[0]}, test data {z_comp_test.shape[0]} -\n\
                        trained with number of clusters = {GIN.n_classes} -with cross validation = {cross_validation}', fontsize = 8)

            plt.legend()
            plt.show()

            if GIN.dataset == 'EMNIST':
                try:
                    os.makedirs(f'{save_dir}\stability_experiments')
                except Exception as e:
                    print("Folder already exist.") 
                plt.savefig(f'{save_dir}\stability_experiments\{method}_informative_name.pdf')
            
            else:
                try:
                    os.makedirs(f'{save_dir}\stability_experiments')
                except Exception as e:
                    print("Folder already exist.") 
                plt.savefig(f'{save_dir}\stability_experiments\{method}_informative_name.pdf')

    else:

        z_ref_val, z_ref_test = get_latent_space_batches(GIN, args, \
            saved_models[0], test_loader, batch=cross_validation)
        z_comp_val, z_comp_test = get_latent_space_batches(GIN, args, \
            saved_models[1], test_loader, batch=cross_validation)

        for method in methods:

            mcc_dims_list = [] 
            mcc_std_dims_list = []

            mcc_dims_list_val = [] 
            mcc_std_dims_list_val = [] 

            for dim in dims:

                mcc_stab_list = [] 
                mcc_stab_list_val = [] 

                for i in range(number_stab_runs):
                    print(method)

                    #################### Apply PCA ##################
                    if "PCA" in method and not "full" in method:
                        print("############## Apply PCA ################")
                        if GIN.dataset == 'EMNIST':
                            PCA_dim = 450 

                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)

                            # print(z_ref_val.shape)

                        else:
                            PCA_dim = GIN.n_dims
                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)
                    
                    if "PCAfull" in method: 
                        print("############## Apply full PCA ################")
                        if GIN.dataset == 'EMNIST':
                            PCA_dim = 784

                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)

                            # print(z_ref_val.shape)

                        else:
                            PCA_dim = GIN.n_dims
                            pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                            z_ref_val = pca_ref.transform(z_ref_val)
                            z_ref_test = pca_ref.transform(z_ref_test)

                            pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                            z_comp_val = pca_comp.transform(z_comp_val)
                            z_comp_test = pca_comp.transform(z_comp_test)


                    ################### Apply Identifyablity metric #

                    X_val = z_ref_val
                    Y_val = z_comp_val
                    X_test = z_ref_test
                    Y_test = z_comp_test

                    # print("Corr(X)")
                    # print(np.round(np.corrcoef(X.T), 2))
                    # print("Corr(Y)")
                    # print(np.round(np.corrcoef(Y.T), 2))

                    # #############################################################################
                    # Canonical (symmetric) PLS

                    # Transform data
                    # ~~~~~~~~~~~~~~
                    try:
                        if "PLSCan" in method:
                            print("############## Apply PLSCanonical ################")
                            plsca = PLSCanonical(n_components=dim, max_iter=2500)
                        elif "CCA" in method:
                            print("############## Apply CCA ################")
                            plsca = CCA(n_components=dim, max_iter=2500)
                        else:
                            print("ERROR: please provide a feature reduction method.")
                            exit(1)
    
                        plsca.fit(X_val, Y_val)
                        X_train_r, Y_train_r = plsca.transform(X_val, Y_val)
                        X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

                        # Compute Mean correlation coefficent over all components
                        mean_cc = 0 
                        mean_cc_val = 0 
                        for k in range(dim):
                            mean_cc += np.corrcoef(X_test_r[:, k], Y_test_r[:, k])[0, 1]
                            mean_cc_val += np.corrcoef(X_train_r[:, k], Y_train_r[:, k])[0, 1]

                        mean_cc /= dim 
                        mean_cc_val /= dim 
                        mcc_stab_list.append(mean_cc)
                        mcc_stab_list_val.append(mean_cc_val)

                    except Exception as e:
                        print(e)
                    
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(e) 

                    print(f"The test MCC value is: {mean_cc}")
            
                mcc_stab = np.mean(np.array(mcc_stab_list))


                mcc_stab  = np.mean(np.array(mcc_stab_list))
                mcc_stab_std = np.std(np.array(mcc_stab_list))

                mcc_stab_val = np.mean(np.array(mcc_stab_list_val))
                mcc_stab_val_std = np.std(np.array(mcc_stab_list_val))

                print(f"The mean correlation coefficient over serveral runs\
                        using {dim} components on validation data is: {mcc_stab_val} +- {mcc_stab_val_std}.")
                print(f"The mean correlation coefficient over serveral models\
                        using {dim} components on test data is: {mcc_stab} +- {mcc_stab_std }.")

                mcc_dims_list.append(mcc_stab) 
                mcc_std_dims_list.append(mcc_stab_std)
                
                mcc_dims_list_val.append(mcc_stab_val)
                mcc_std_dims_list_val.append(mcc_stab_val_std)
            
            mcc_dims_list = np.array(mcc_dims_list)
            mcc_std_dims_list = np.array(mcc_std_dims_list)

            mcc_dims_list_val = np.array(mcc_dims_list_val)
            mcc_std_dims_list_val = np.array(mcc_std_dims_list_val)

            plt.plot(dims, mcc_dims_list, label=f'{method} - test')
            plt.xlabel("final dimension")
            plt.ylabel("MCC")
            plt.fill_between(dims, mcc_dims_list - mcc_std_dims_list, mcc_dims_list + mcc_std_dims_list, alpha=0.2)
            # plt.plot(dims, mcc_dims_list_val,  label='val')
            # plt.fill_between(dims, mcc_dims_list_val - mcc_std_dims_list_val, mcc_dims_list_val + mcc_std_dims_list_val, color='r', alpha=0.2)
            if "PCA" in method:
                plt.title(f'MCC stability - GLOW - Feature reduction method: {method} - PCA dim = {PCA_dim} -\n \
                    final dim = {dim} - Dataset {GIN.dataset} - number of validation data {z_comp_val.shape[0]}, test data {z_comp_test.shape[0]} -\n\
                        trained with number of clusters = {GIN.n_classes} -with cross validation = {cross_validation}', fontsize = 10)

            else:
                plt.title(f'MCC stability - GLOW - Feature reduction method: {method} - \n\
                    final dim = {dim} - Dataset {GIN.dataset} - number of validation data {z_comp_val.shape[0]}, test data {z_comp_test.shape[0]} -\n\
                        trained with number of clusters = {GIN.n_classes} -with cross validation = {cross_validation}', fontsize = 10)

            plt.legend()
            plt.show()

            if GIN.dataset == 'EMNIST':
                try:
                    os.makedirs(f'{save_dir}\stability_experiments')
                except Exception as e:
                    print("Folder already exist.") 
                plt.savefig(f'{save_dir}\stability_experiments\{method}_informative_name.pdf')


            else:
                try:
                    os.makedirs(f'{save_dir}\stability')
                except Exception as e:
                    print("Folder already exist.") 
                plt.savefig(f'{save_dir}\stability\MCC_values_{GIN.n_classes}_cluster.pdf')



def evaluate_stability_many_data(args, GIN, save_dir, cross_validation=False):

    print("############## Stability Evaluation ################")

    number_stab_runs = 6
    if GIN.dataset == 'EMNIST':
        # dims = [25]
        dim = 25
        batch_sizes = [200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000 ] # , 1000, 1500, 3000, 5000 ]
        # batch_size = n = 5000
        n_batches = 6
        # test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir, shuffle=False)

    else:
        # dims = [GIN.n_dims]
        dim = GIN.n_dims
        n_batches = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f)) and '.pt' in f ]

    methods = ["PCA+PLSCan", "PCAfull+PLSCan", "PLSCan", "PCA+CCA", "CCA" ]
    # methods = ["CCA", "PLSCan", "PCAfull+PLSCan"]

    print(f"Using models {saved_models[0]} as reference model and {saved_models[1]} model to compare with.")

    for method in methods:

        mcc_dims_list = [] 
        mcc_std_dims_list = []

        mcc_dims_list_val = [] 
        mcc_std_dims_list_val = [] 

        for batch_size in batch_sizes:
            print(f"Apply Cross Validation for batch size: {batch_size}")
            # test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir, shuffle=False)

            z_ref_set = get_latent_space_batches(GIN, args, \
                saved_models[0], batch_size=batch_size,  num_batches=n_batches, batch=cross_validation)
            z_comp_set = get_latent_space_batches(GIN, args, \
                saved_models[1], batch_size=batch_size, num_batches=n_batches, batch=cross_validation)


            mcc_stab_list = [] 
            mcc_stab_list_val = [] 

            # cross validation

            for k in range(len(z_ref_set)):

                z_ref_test = z_ref_set[k]
                z_comp_test = z_comp_set[k]
                z_ref_val = np.concatenate([z_ref_set[m] for m in range(len(z_ref_set)) if m != k], axis=0)
                z_comp_val = np.concatenate([z_comp_set[m] for m in range(len(z_ref_set)) if m != k], axis=0)

                # print(z_comp_val.shape)

                print(method)

                #################### Apply PCA ##################
                if "PCA" in method and not "full" in method: 
                    print("############## Apply PCA ################")
                    if GIN.dataset == 'EMNIST':
                        PCA_dim = 450 

                        pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                        z_ref_val = pca_ref.transform(z_ref_val)
                        z_ref_test = pca_ref.transform(z_ref_test)

                        pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                        z_comp_val = pca_comp.transform(z_comp_val)
                        z_comp_test = pca_comp.transform(z_comp_test)

                        # print(z_ref_val.shape)

                    else:
                        PCA_dim = GIN.n_dims
                        pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                        z_ref_val = pca_ref.transform(z_ref_val)
                        z_ref_test = pca_ref.transform(z_ref_test)

                        pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                        z_comp_val = pca_comp.transform(z_comp_val)
                        z_comp_test = pca_comp.transform(z_comp_test)

                if "PCAfull" in method: 
                    print("############## Apply full PCA ################")
                    if GIN.dataset == 'EMNIST':
                        PCA_dim = 784

                        pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                        z_ref_val = pca_ref.transform(z_ref_val)
                        z_ref_test = pca_ref.transform(z_ref_test)

                        pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                        z_comp_val = pca_comp.transform(z_comp_val)
                        z_comp_test = pca_comp.transform(z_comp_test)

                        # print(z_ref_val.shape)

                    else:
                        PCA_dim = GIN.n_dims
                        pca_ref = PCA(n_components=PCA_dim).fit(z_ref_val)
                        z_ref_val = pca_ref.transform(z_ref_val)
                        z_ref_test = pca_ref.transform(z_ref_test)

                        pca_comp = PCA(n_components=PCA_dim).fit(z_comp_val)
                        z_comp_val = pca_comp.transform(z_comp_val)
                        z_comp_test = pca_comp.transform(z_comp_test)

                print(f"the dimension of the latent space is: {z_ref_val.shape[1]}")
                print(f"the number of validation data is: {z_ref_val.shape[0]}")
                ################### Apply Identifyablity metric #

                X_val = z_ref_val
                Y_val = z_comp_val
                X_test = z_ref_test
                Y_test = z_comp_test

                # print("Corr(X)")
                # print(np.round(np.corrcoef(X.T), 2))
                # print("Corr(Y)")
                # print(np.round(np.corrcoef(Y.T), 2))

                # #############################################################################
                # Canonical (symmetric) PLS

                # Transform data
                # ~~~~~~~~~~~~~~
                try:
                    if "PLSCan" in method:
                        print("############## Apply PLSCanonical ################")
                        plsca = PLSCanonical(n_components=dim, max_iter=2500)
                    elif "CCA" in method:
                        print("############## Apply CCA ################")
                        plsca = CCA(n_components=dim, max_iter=2500)
                    else:
                        print("ERROR: please provide a feature reduction method.")
                        exit(1)

                    plsca.fit(X_val, Y_val)
                    X_train_r, Y_train_r = plsca.transform(X_val, Y_val)
                    X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

                    # Compute Mean correlation coefficent over all components
                    mean_cc = 0 
                    mean_cc_val = 0 
                    for k in range(dim):
                        # print(np.corrcoef(X_test_r[:, k], Y_test_r[:, k]))
                        # print(np.corrcoef(X_test_r[:, k], Y_test_r[:, k]).shape)
                        mean_cc += np.corrcoef(X_test_r[:, k], Y_test_r[:, k])[0, 1]
                        mean_cc_val += np.corrcoef(X_train_r[:, k], Y_train_r[:, k])[0, 1]
                    print("the len vs dim is:", len(range(dim)), dim)

                    mean_cc /= dim 
                    mean_cc_val /= dim 
                    mcc_stab_list.append(mean_cc)
                    mcc_stab_list_val.append(mean_cc_val)

                except Exception as e:
                    print(e)
                
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(e) 

                print(f"The test MCC value is: {mean_cc}")
                print(f"The validation MCC value is: {mean_cc_val}")
        
            mcc_stab = np.mean(np.array(mcc_stab_list))


            mcc_stab  = np.mean(np.array(mcc_stab_list))
            mcc_stab_std = np.std(np.array(mcc_stab_list))

            mcc_stab_val = np.mean(np.array(mcc_stab_list_val))
            mcc_stab_val_std = np.std(np.array(mcc_stab_list_val))

            print(f"The mean correlation coefficient over serveral runs\
                    using {dim} components on validation data is: {mcc_stab_val} +- {mcc_stab_val_std}.")
            print(f"The mean correlation coefficient over serveral models\
                    using {dim} components on test data is: {mcc_stab} +- {mcc_stab_std }.")

            mcc_dims_list.append(mcc_stab) 
            mcc_std_dims_list.append(mcc_stab_std)
            
            mcc_dims_list_val.append(mcc_stab_val)
            mcc_std_dims_list_val.append(mcc_stab_val_std)
        
        mcc_dims_list = np.array(mcc_dims_list)
        mcc_std_dims_list = np.array(mcc_std_dims_list)

        mcc_dims_list_val = np.array(mcc_dims_list_val)
        mcc_std_dims_list_val = np.array(mcc_std_dims_list_val)

        plt.plot(batch_sizes, mcc_dims_list, label=f'{method}')
        plt.xlabel("batch_size")
        plt.ylabel("MCC")
        plt.fill_between(batch_sizes, mcc_dims_list - mcc_std_dims_list, mcc_dims_list + mcc_std_dims_list, alpha=0.2)
        # plt.plot(dims, mcc_dims_list_val, 'r-', label='val')
        # plt.fill_between(dims, mcc_dims_list_val - mcc_std_dims_list_val, mcc_dims_list_val + mcc_std_dims_list_val, color='r', alpha=0.2)
        if "PCA" in method:
            plt.title(f'MCC stability on Test data - GLOW - Feature reduction method: {method} - PCA dim = {PCA_dim} - \n\
                final dim = {dim} - Dataset {GIN.dataset} - number of batches {n_batches} -\n\
                trained with number of clusters = {GIN.n_classes} - with cross validation = {cross_validation}', fontsize = 10)

        else:
            plt.title(f'MCC stability on Test data - GLOW - Feature reduction method: {method} - \n\
                final dim = {dim} - Dataset {GIN.dataset} - number of batches {n_batches} -\n\
                trained with number of clusters = {GIN.n_classes} -with cross validation = {cross_validation}', fontsize = 10)

        plt.legend()
        plt.show()

        if GIN.dataset == 'EMNIST':
            try:
                os.makedirs(f'{save_dir}\stability_experiments')
            except Exception as e:
                print("Folder already exist.") 
            plt.savefig(f'{save_dir}\stability_experiments\{method}_informative_name.pdf')


def get_latent_space_batches(GIN, args, model_path, test_loader=None , batch=False, batch_size=5000, num_batches=6, val_batches=5):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GIN.to(device)
    data = torch.load(model_path)
    model.load_state_dict(data['model'])
    model.to(device)
    model.eval()

    # print(f"The parameters of model: {model_path} are: \n {model.pi_c} \n and \n {model.mu_c}." )

    if GIN.dataset == 'EMNIST':
        assert batch_size < 5001, "batch size should be not larger than 5000."

        test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir, shuffle=False)

        z_set = []
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx < num_batches:
                data_val = data.to(device)
                z_batch = model(data.to(device))[0].cpu().detach() 
                z_set.append(z_batch)

        del model

        if not batch:

            z_val = np.concatenate(z_set[:(val_batches-1)], axis=0)
            z_test = np.concatenate(z_set[(val_batches-1):], axis=0)
            return z_val, z_test
        else:
            return z_set

    # elif GIN.dataset == '10d':
    #     n = args.n_data_points
    #     data_val = GIN.data.to(device)
    #     print(f"The number of used clusters is: {GIN.n_classes} ")

    #     z_val_1 = model(data_val[: n // 3 ])[0].cpu().detach() 
    #     z_val_2 = model(data_val[n // 3 : int(n - n // 3) ])[0].cpu().detach() 
    #     z_val = np.append(z_val_1, z_val_2, axis=0)
    #     z_test = model(data_val[ int(n - n // 3) : ])[0].cpu().detach() 
    #     del model

    #     return z_val, z_test

    elif GIN.dataset == '10d':
        if batch_size*num_batches > args.n_data_points:
            print(f"ERROR: The batch size times number of batches (= {batch_size*num_batches}) \
                exceeds total number of datapoints (= {args.n_data_points}).\n \
                    Abort further execution. ")
            exit(1)

        n = batch_size
        data_val = GIN.data.to(device)
        print(f"The number of used clusters is: {GIN.n_classes} ")

        z_set = []
        for i in range(num_batches):
            data = data_val[i*batch_size: (i+1)*batch_size ]
            z_batch = model(data.to(device))[0].cpu().detach() 
            z_set.append(z_batch)
        
        print(z_set)
        del model

        if not batch:
            z_val = np.concatenate(z_set[:(val_batches-1)], axis=0)
            z_test = np.concatenate(z_set[(val_batches-1):], axis=0)
            return z_val, z_test
        else:
            return z_set

        
    else:
        print("ERROR: please use a valid dataset.")
        exit(1)


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
