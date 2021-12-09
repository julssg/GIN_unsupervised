from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import pandas as pd
from scipy.stats import special_ortho_group
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.decomposition import PCA
from sklearn.base import clone 
from data import make_dataloader_emnist


def mcc_evaluation(
    base_model, 
    args, 
    save_dir,
    cross_validation=False, 
    plot=False
    ):
    '''This functions has input parameters:
    base_model: GIN network architecture defined in model.py.
    args: set of arguments set in command line while training model.
    save_dir: path to saved models which latent spaces will be compared, 
                all files with .pt in this folder will be assumed as final model checkpoints
    cross_validation: bool, whether to use or not to use cross validation
    evaluation: which form of evaluation to use, choose from ['cross_models', 'cross_data', 'cross_dims']
        'cross_models': in model_path are at least 3 trained models, and the MCC value is averaged over 
                        each pairwise comparison
        'cross_datas': check the stability of calculated MCC by comparing trained models 
                        for different sizes of validation and test data.
        'cross_dims': check the stability of calculated MCC by comparing trained models 
                        for different final dimensions.
    plot: bool, whether to plot or not to plot results.
    '''
    # assert evaluation in ['cross_models', 'cross_datas', 'cross_dims'], \
    # "choose a evaluation from ['cross_models', 'cross_datas', 'cross_dims']"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    methods = ["CCA" ,"PLSCan", "PCA+PLSCan"]
    saved_models = [join(save_dir, f) for f in listdir(save_dir) if isfile(join(save_dir, f)) and '.pt' in f ] 
    num_models = len(saved_models)

    if base_model.dataset == 'EMNIST':
        n_train_data = "default"
        batch_size = 5000
        num_batches = 6
        val_batches = 5
        dims = np.arange(15, 35, 2)
    elif base_model.dataset == '10d':
        n_train_data = (load(base_model, saved_models[0], device)).data.to(device).shape[0]
        print(n_train_data)
        batch_size = 100
        val_batches = 3
        num_batches = 4
        dims = [5, 10] # np.arange(1 , GIN.n_dims+1 , 1)
        print(f"The number of used clusters is: {base_model.n_classes} ")
    else:
        print(f"ERROR: {base_model.dataset} as dataset is not defined.")

    print(f"##############  Starting Identifiability Evaluation ################")

    dict = {'dimension':[],
        'mcc_value':[],
        'data':[],
        'n_training_points':[],
        'method':[]
       }
    df = pd.DataFrame(dict)

    for method in methods:
        print(f"##############  Using method: {method} ################")
        for dim in dims:
            mcc_val_cross_models = []
            mcc_test_cross_models = []
            for i in range(num_models):
                for j in range(num_models):
                    if (j == i) or (j<i):
                        continue

                    if cross_validation:
                        z_ref_set = get_latent_space_samples(base_model, args, device, \
                            saved_models[i], for_cross_validation=True, batch_size=batch_size, \
                            num_batches=num_batches, val_batches=val_batches)
                        z_comp_set = get_latent_space_samples(base_model, args, device, \
                            saved_models[j], for_cross_validation=True, batch_size=batch_size, \
                            num_batches=num_batches, val_batches=val_batches)
                    else:
                        z_ref_set = get_latent_space_samples(base_model, args, device, \
                            saved_models[i], for_cross_validation=False, batch_size=batch_size, \
                            num_batches=num_batches, val_batches=val_batches)
                        z_comp_set  =get_latent_space_samples(base_model, args, device, \
                            saved_models[j], for_cross_validation=True, batch_size=batch_size, \
                            num_batches=num_batches, val_batches=val_batches)

                    mcc_val_cross_validation = []
                    mcc_test_cross_validation = []
                    for k in range(len(z_ref_set)):
                        # if cross validation=False, then k is in [0,1], I only want one interation, so I force 
                        # to skip k == 0, then 
                        # if cross_validation==false: 1 iteration with test_data=set[1] & val_data=set[0]
                        # if cross_validation==true: len(z_ref_set)-1 iteration with test_data in [set[1:len(z_ref_set)] 
                        if k == 0:
                            continue
                        z_ref_test = z_ref_set[k]
                        z_comp_test = z_comp_set[k]
                        z_ref_val = np.concatenate([z_ref_set[m] for m in range(len(z_ref_set)) if m != k], axis=0)
                        z_comp_val = np.concatenate([z_comp_set[m] for m in range(len(z_ref_set)) if m != k], axis=0)

                        x_val, y_val, x_test, y_test = apply_method(method, z_ref_test, z_comp_test, \
                            z_ref_val, z_comp_val, args.n_classes)

                        mcc_val_cross_validation.append(compute_mcc(x_val, y_val))
                        mcc_test_cross_validation.append(compute_mcc(x_test, y_test))

                    mcc_val = np.mean(np.array(mcc_val_cross_validation))
                    mcc_test = np.mean(np.array(mcc_test_cross_validation))
                    mcc_val_cross_models.append(mcc_val)
                    mcc_test_cross_models.append(mcc_test)
                    if base_model.dataset == 'EMNIST':
                        df.loc[len(df.index)] = [f'{dim}', mcc_val, 'val', f'{n_train_data}', method] 
                        df.loc[len(df.index)] = [f'{dim}', mcc_test, 'test',  f'{n_train_data}', method]
                    elif base_model.dataset == '10d':
                        df.loc[len(df.index)] = [f'{dim}', mcc_val, 'val', f'{(n_train_data/1000):.1f} k', method] 
                        df.loc[len(df.index)] = [f'{dim}', mcc_test, 'test',  f'{(n_train_data/1000):.1f} k', method]

            print(f"The mean correlation coefficient over serveral models\
                using {dim} components an method {method} on validation data is: \n \
                {np.mean(np.array(mcc_val_cross_models))} +- {np.std(np.array(mcc_val_cross_models))}.")
            print(f"The mean correlation coefficient over serveral models\
                using {dim} components an method {method}  on test data is: \n \
                {np.mean(np.array(mcc_test_cross_models))} +- {np.std(np.array(mcc_test_cross_models))}.")

    return df


def apply_method(method, z_ref_test, z_comp_test, z_ref_val, z_comp_val, n_classes):
    ''' Apply method for dimension reduction and maximizaion of cross correlation.
    Input:
        latent spaces of reference and comparison model, each with validation 
        samples to fit the method, and test samples to evaluate generalization of 
        method.
    Return:
        latent space samples which only have most cross correlated 
        classes/dimensions (n_classes) left. 
    '''
    try:
        if "PCA" in method:
            print("############## Apply PCA ################")
            if z_ref_test.shape[1] == 784:
                pca_dim = 450
            elif z_ref_test.shape[1] == 10:
                pca_dim = 10
            else:
                print(f"ERROR: dimension of latent space is {z_ref_test.shape[1]}, \
                    expected 10 or 784.")
                exit(1)
            pca = PCA(n_components=pca_dim)
            pca.fit(z_ref_val)
            z_ref_val = pca.transform(z_ref_val)
            z_ref_test = pca.transform(z_ref_test)

            pca = clone(pca)
            pca.fit(z_comp_val)
            z_comp_val = pca.transform(z_comp_val)
            z_comp_test = pca.transform(z_comp_test)

        if "PLSCan" in method:
            print("############## Apply PLSCanonical ################")
            plsca = PLSCanonical(n_components=n_classes, max_iter=2500)
        elif "CCA" in method:
            print("############## Apply CCA ################")
            plsca = CCA(n_components=n_classes, max_iter=2500)
        else:
            print("ERROR: please provide a feature reduction method.")
            exit(1)

        #### transform latent spaces:
        plsca.fit(z_ref_val, z_comp_val)
        x_val, y_val = plsca.transform(z_ref_val, z_comp_val)
        x_test, y_test = plsca.transform(z_ref_test, z_comp_test)
    
    except Exception as e:
        print(e)
        x_val, y_val = None, None
        x_test, y_test = None, None

    return x_val, y_val, x_test, y_test


def compute_mcc(x, y):
    ''' 
    compute the correlation between each pairwise set of samples and average over all samples.  
    x.shape = y.shape = [n_samples, n_features]
    No dimensionality reduction happens in this function.
    '''
    if not x:
        return
    else:
        mean_cc = np.sum([ np.corrcoef(x[:, k], y[:, k])[0, 1] for k in range(x.shape[1])] ) / x.shape[1]
        return mean_cc


def get_latent_space_samples(
    base_model, 
    args, 
    device, 
    model_path, 
    for_cross_validation=False, 
    batch_size=5000, 
    num_batches=6, 
    val_batches=5
    ):
    '''
    base_model: GIN as defined with args in emnist.py or artifical_data.py,
    args: passed args as defined in emnist.py or artifical_data.py,
    device: 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path: path to saved models which latent spaces will be compared,
    for_cross_validation: bool, whether data should be prepeared for cross validation or concatenated
    batch_size: in the case where 'for_cross_validation' is True, it defines the size of each compartment
    num_batches: total number of batches = val_batches + test_batches
    val_batches: number of validation batches
     '''

    model = load(base_model, model_path, device)

    if base_model.dataset == 'EMNIST':
        assert batch_size < 5001, "batch size should not be larger than 5000."

        test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir, shuffle=False)

        z_set = []
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx < num_batches:
                data_val = data.to(device)
                z_batch = model(data.to(device))[0].cpu().detach() 
                z_set.append(z_batch)

        del model

        if not for_cross_validation:

            z_val = np.concatenate(z_set[:(val_batches-1)], axis=0)
            z_test = np.concatenate(z_set[(val_batches-1):], axis=0)
            return [z_val, z_test]
        else:
            return z_set

    elif base_model.dataset == '10d':
        # Use the data from the base model, which were not used for training for validation
        data_val = base_model.data.to(device)

        if batch_size*num_batches > data_val.shape[0]:
            print(f"ERROR: The batch size times number of batches (= {batch_size*num_batches}) \
                exceeds total number of datapoints (= {data_val.shape[0]}).\n \
                    Abort further execution. ")
            exit(1)
        z_set = []
        for i in range(num_batches):
            data = data_val[i*batch_size: (i+1)*batch_size ]
            z_batch = model(data.to(device))[0].cpu().detach() 
            z_set.append(z_batch)
        del model

        if not for_cross_validation:
            z_val = np.concatenate(z_set[:(val_batches-1)], axis=0)
            z_test = np.concatenate(z_set[(val_batches-1):], axis=0)
            return [z_val, z_test]
        else:
            return z_set

    else:
        print("ERROR: please use a valid dataset.")
        exit(1)


def load(base_model, model_path, device):
    model = base_model.to(device)
    data = torch.load(model_path)
    model.load_state_dict(data['model'])
    model.to(device)
    model.eval()
    return model


def test_metric():
    '''
    This is a standalone function to test the different metrics and their behaviour
    for low vs high dimensional data, and for comparing different cases:
    (latent_1, latent_1)
    (latent_1, random rotation(latent_1))
    (latent_1, latent_2).
    '''

    methods = ["PLSCan", "CCA", "PCA+PLSCan" ]
    data_type = "high_dim"       # choose from ["low_dim", "high_dim"] 

    assert data_type in ["low_dim", "high_dim"], "choose data_type from ['low_dim', 'high_dim']"

    if data_type == "low_dim":
        n_clusters = 5
        n_data_points = 200
        latent_means = torch.rand(n_clusters, 2)*10 - 5         # in range (-5, 5)
        latent_stds  = torch.rand(n_clusters, 2)*2.5 + 0.5      # in range (0.5, 3)
        labels = torch.randint(n_clusters, size=(n_data_points,))
        latent = latent_means[labels] + torch.randn(n_data_points, 2)*latent_stds[labels]
        latent_init = torch.cat([latent, torch.randn(n_data_points, 8)*1e-2], 1).numpy()
        rotation_matrix = special_ortho_group.rvs(10) 

        ### alternative latent space:
        latent_alt_means = torch.rand(n_clusters, 2)*10 - 5         # in range (-5, 5)
        latent_alt_stds  = torch.rand(n_clusters, 2)*2.5 + 0.5      # in range (0.5, 3)
        labels_alt = torch.randint(n_clusters, size=(n_data_points,))
        latent_alt = latent_alt_means[labels_alt] + torch.randn(n_data_points, 2)*latent_alt_stds[labels_alt]
        latent_alt_init = torch.cat([latent_alt, torch.randn(n_data_points, 8)*1e-2], 1).numpy()
    
    elif data_type == "high_dim":
        n_clusters = 40 
        n_data_points = 20000
        latent_means = torch.rand(n_clusters, 450)*10 - 5         # in range (-5, 5)
        latent_stds  = torch.rand(n_clusters, 450)*2.5 + 0.5      # in range (0.5, 3)
        # print(f"The latents have shape: {latent_means.shape}.")
        labels = torch.randint(n_clusters, size=(n_data_points,))
        latent = latent_means[labels] + torch.randn(n_data_points, 450)*latent_stds[labels]
        latent_init = torch.cat([latent, torch.randn(n_data_points, 334)*1e-2], 1).numpy()
        rotation_matrix = special_ortho_group.rvs(784)

        ### alternative latent space:
        latent_alt_means = torch.rand(n_clusters, 450)*10 - 5         # in range (-5, 5)
        latent_alt_stds  = torch.rand(n_clusters, 450)*2.5 + 0.5      # in range (0.5, 3)
        labels_alt = torch.randint(n_clusters, size=(n_data_points,))
        latent_alt = latent_alt_means[labels_alt] + torch.randn(n_data_points, 450)*latent_alt_stds[labels]
        latent_alt_init = torch.cat([latent_alt, torch.randn(n_data_points, 334)*1e-2], 1).numpy()

    latent_rotated_init = np.matmul(rotation_matrix,  latent_init.T).T
    print(f"The latents have shape: {latent_init.shape}, and the rotated latents have shape {latent_rotated_init.shape}.")

    for method in methods:
        latent = latent_init
        latent_alt = latent_alt_init
        latent_rotated = latent_rotated_init

        if "PCA" in method:
            print("############## Apply PCA ################")
            if data_type == "low_dim":
                pca_dim = rotation_matrix.shape[0]
            elif data_type == "high_dim":
                pca_dim = 450
            pca = PCA(n_components=pca_dim)
            pca.fit(latent)
            latent = pca.transform(latent)

            pca = clone(pca)
            pca.fit(latent_rotated)
            latent_rotated = pca.transform(latent_rotated)

            pca = clone(pca)
            pca.fit(latent_alt)
            latent_alt = pca.transform(latent_alt)

        if "PLSCan" in method:
            print("############## Apply PLSCanonical ################")
            plsca = PLSCanonical(n_components=n_clusters, max_iter=2500)
        elif "CCA" in method:
            print("############## Apply CCA ################")
            plsca = CCA(n_components=n_clusters, max_iter=2500)
        else:
            print("ERROR: please provide a feature reduction method.")
            exit(1)

        #### Compare latent original with itself:

        plsca.fit(latent, latent)
        latent_transformed_x, latent_transformed_y = plsca.transform(latent, latent)

        # print(f"A transformed latent space sample looks like this: {latent_transformed_x[0]}. \n \
        #    The same sample as y in fit transforms to {latent_transformed_y[0]}.")
        mean_cc = compute_mcc(latent_transformed_x, latent_transformed_y)
        print(f"The mean correlation coefficient when using the same set of latents and method {method}: \n \
             MCC = {mean_cc} \n \
        Expect value == 1 - statistical variance due to approximations in dimension reduction.")

        #### Compare latent original with latent rotated:
        
        plsca = clone(plsca)
        plsca.fit(latent, latent_rotated)
        latent_transformed_x, latent_rotated_transformed_y = plsca.transform(latent, latent_rotated)

        # print(f"A transformed latent space sample looks like this: {latent_transformed_x[0]}. \n \
        #     The latent space rotated sample as y in fit transforms to {latent_rotated_transformed_y [0]}.")
        mean_cc = compute_mcc(latent_transformed_x, latent_rotated_transformed_y)
        print(f"The mean correlation coefficient when using the rotated set of latents and method {method}: \n \
            MCC = {mean_cc}\n \
        Expect value == 1 - statistical variance due to approximations in dimension reduction + transformation.")

        #### Compare latent original with random latent alternative:
        
        plsca = clone(plsca)
        plsca.fit(latent, latent_alt)
        latent_transformed_x, latent_alt_transformed_y = plsca.transform(latent, latent_alt)

        # print(f"A transformed latent space sample looks like this: {latent_transformed_x[0]}. \n \
        #     The alternative latent space sample as y in fit transforms to {latent_alt_transformed_y [0]}.")
        mean_cc = compute_mcc(latent_transformed_x, latent_alt_transformed_y)
        print(f"The mean correlation coefficient when using the random set of latents and method {method}: \n \
            MCC = {mean_cc}. \n \
        Expect low value. ")


if __name__ == '__main__':
    test_metric()