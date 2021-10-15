from os import listdir
from os.path import isfile, join
import torch
from sklearn.cross_decomposition import CCA
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
        batch_size = 2000
        test_loader  = make_dataloader_emnist(batch_size=batch_size, train=False, root_dir=args.data_root_dir)
        dims = [200]

        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx < 1:
                data_val = data.to(device)
    
    else:
        data_val = GIN.data.to(device)
        dims = [10]

    score_values_list =[]

    for dim in dims:
        for i in range(num_runs):
            for j in range(num_runs):
                if j != i:
                    print(f"Using models {saved_models[i]} as reference model and {saved_models[j]} model to compare with.")
                    
                    model = GIN.to(device)
                    data = torch.load(saved_models[i])
                    model.load_state_dict(data['model'])
                    model.to(device)
                    model.eval()

                    z, logdet_J = model.net(data_val)
                    z_ref_val = z.cpu().detach()

                    del model

                    model = GIN
                    data = torch.load(saved_models[j])
                    model.load_state_dict(data['model'])
                    model.to(device)
                    model.eval()
                    z, logdet_J = model.net(data_val)
                    z_comp_val = z.cpu().detach()

                    del model

                    # test different cca evaluation: 
                    # z_ref_val = np.random.rand(batch_size,784)
                    # z_comp_val = np.random.rand(batch_size,784)
                    # print((z_ref_val[:int(batch_size/2)]).shape)

                    cca_dim = dim
                    cca = CCA(n_components=cca_dim, max_iter=500)
                    cca.fit(z_ref_val[:int(batch_size/2)], z_comp_val[:int(batch_size/2)])
                    res_out = cca.transform(z_ref_val[:int(batch_size/2)], z_comp_val[:int(batch_size/2)])
                    mcc_weak_out = mean_corr_coef(res_out[0], res_out[1], method = 'spearman')
                    res_in = cca.transform(z_ref_val[int(batch_size/2):], z_comp_val[int(batch_size/2):])
                    mcc_weak_in = mean_corr_coef(res_in[0], res_in[1])
                    print('mcc weak in: ', mcc_weak_in, ' --- ccadim = ', cca_dim)
                    print('mcc weak out: ', mcc_weak_out, ' --- ccadim = ', cca_dim)

                    cca = CCA(n_components=cca_dim, max_iter=500)
                    cca.fit(z_comp_val[:int(batch_size/2)], z_ref_val[:int(batch_size/2)])
                    res_out = cca.transform(z_comp_val[:int(batch_size/2)], z_ref_val[:int(batch_size/2)])
                    mcc_weak_out = mean_corr_coef(res_out[0], res_out[1])
                    res_in = cca.transform(z_comp_val[int(batch_size/2):], z_ref_val[int(batch_size/2):])
                    mcc_weak_in = mean_corr_coef(res_in[0], res_in[1])
                    print('mcc weak in: ', mcc_weak_in, ' --- ccadim = ', cca_dim)
                    print('mcc weak out after swapping: ', mcc_weak_out, ' --- ccadim = ', cca_dim)


                    exit(1)

                    # reference:
                    if i == 0:
                        try:
                            print(f"Fitting reference CCA with {dim} components....")
                            cca = CCA(n_components=dim)
                            cca.fit(z_ref_val, z_ref_val)
                            score_ref_val =  cca.score(z_ref_val , z_ref_val)

                            cca = CCA(n_components=dim)
                            cca.fit(z_comp_val, z_comp_val)
                            score_comp_val =  cca.score(z_comp_val , z_comp_val)
                            print(f"The reference scores for model {i} is {score_ref_val} and for {j} is {score_comp_val} ")

                        except Exception as e:
                            print(e)
                    else: 
                        score_ref_val = "see above"
                        score_comp_val = "see above"

                    
                    try:
                        print(f"Fitting CCA with {dim} components....")
                        cca = CCA(n_components=dim, max_iter=5000)
                        cca.fit(z_ref_val, z_comp_val)
                        print(f"Evaluating CCA with {dim} components for latent spaces....")
                        score_val = cca.score(z_ref_val , z_comp_val)
                        print(f"The validation score for models {i} and {j} for cca in latent space is: {score_val}")
                        try:
                            norm_score_val = score_val**2/(score_ref_val*score_comp_val)
                        except Exception as e:
                            print(e)
                            norm_score_val = None
                        print(f"The validation score including the reference value for models {i} and {j} for cca in latent space is: {norm_score_val}")
                        score_values_list.append(score_val)

                        with open(f'{save_dir}\cca_score_{dim}.csv', 'a') as csvfile:
                            filewriter = csv.writer(csvfile, delimiter=',',
                                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            
                            filewriter.writerow([saved_models[i], saved_models[j], score_ref_val, score_comp_val, score_val, norm_score_val ])

                        del cca
                        
                    except Exception as e:
                        print(e)
                        print("Continue with next model comparison.")
        
        mean_score_value = sum(score_values_list)/len(score_values_list)

        with open(f'{save_dir}\cca_score_{dim}.csv', 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['mean value',' ' , ' ' , ' ' , mean_score_value, ' ' ])


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
