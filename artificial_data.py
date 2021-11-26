import argparse
import torch
import os
from collections import OrderedDict
from model import GIN
from evaluate import cca_evaluation, evaluate_stability

parser = argparse.ArgumentParser(description='Artificial data experiments (2d data embedded in 10d) with GIN.')
parser.add_argument('--n_clusters', type=int, default=5,
                    help='Number of components in gaussian mixture (default 5)')
parser.add_argument('--n_data_points', type=int, default=10000,
                    help='Number of data points in artificial data set (default 10000)')
parser.add_argument('--n_epochs', type=int, default=80,
                    help='Number of training epochs (default 80)')
parser.add_argument('--epochs_per_line', type=int, default=5,
                    help='Print a new line after this many epochs (default 5)')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Learn rate (default 1e-2)')
parser.add_argument('--lr_schedule', nargs='+', type=int, default=[50],
                    help='Learn rate schedule (decrease lr by factor of 10 at these epochs, default [50]). \
                            Usage example: --lr_schedule 20 40')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size (default 1000)')
parser.add_argument('--incompressible_flow', type=int, default=1,
                    help='Use an incompressible flow (GIN) (1, default) or compressible flow (GLOW) (0)')
parser.add_argument('--empirical_vars', type=int, default=1,
                    help='Estimate empirical variables (means and stds) for each batch (1, default) or learn them along \
                            with model weights (0)')
parser.add_argument('--unsupervised', type=int, default=0,
                    help='State whether model should be trained supervised (0, default) or unsupervised \
                            with leared clustering in latent space (1)')
parser.add_argument('--evaluate', type=int, default=0,
                    help='State whether model should be evaluated (1) or not (0, default)')
parser.add_argument('--init_identity', type=int, default=1,
                    help='Initialize the network as the identity (1, default) or not (0)')

args = parser.parse_args()
assert args.incompressible_flow in [0,1], 'Argument should be 0 or 1'
assert args.empirical_vars in [0,1], 'Argument should be 0 or 1'
assert args.unsupervised in [0,1], 'Argument should be 0 or 1'
assert args.init_identity in [0,1], 'Argument should be 0 or 1'
assert args.evaluate in [0,1], 'Argument should be 0 or 1'

model_origin = GIN(dataset='10d', 
            n_classes=args.n_clusters, 
            n_data_points=args.n_data_points, 
            n_epochs=args.n_epochs, 
            epochs_per_line=args.epochs_per_line, 
            lr=args.lr, 
            lr_schedule=args.lr_schedule, 
            batch_size=args.batch_size, 
            incompressible_flow=args.incompressible_flow, 
            empirical_vars=args.empirical_vars, 
            unsupervised=args.unsupervised,
            init_identity=args.init_identity, 
            save_frequency=args.n_epochs)

# save a inital version of the model, to train the model multiple times on the same artifical data set (initialization is the same)

state_dict = OrderedDict((k,v) for k,v in model_origin.state_dict().items() if not k.startswith('net.tmp_var'))
os.makedirs(os.path.join(model_origin.save_dir, 'model_save'))
os.makedirs(os.path.join(model_origin.save_dir, 'figures'))
trained_models_folder = os.path.join(model_origin.save_dir, 'model_save', 'trained_final')
os.makedirs(trained_models_folder)
init_model_path = os.path.join(model_origin.save_dir, 'model_save', 'init.pt')
torch.save({'model': state_dict}, init_model_path )

for i in range(2):
        model = model_origin
        data = torch.load(init_model_path)
        model.load_state_dict(data['model'])
        model.to(model.device)
        model.n_classes = 5 
        model.initialize()   # reinit to have different seeds
        model.train_model()
        trained_models_cluster_folder = os.path.join(trained_models_folder, f"{model.n_classes}_clusters")
        try:
                os.makedirs(trained_models_cluster_folder)
        except Exception as e:
                print(e)
        trained_model_path = os.path.join(trained_models_cluster_folder, f'trained_model_{i}.pt')
        state_dict = OrderedDict((k,v) for k,v in model.state_dict().items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, trained_model_path)
        del model
        

if args.evaluate:
        save_dir = trained_models_cluster_folder
        # cca_evaluation(args, model_origin, save_dir)
        evaluate_stability(args, model_origin, save_dir, cross_validation=True)

# for i in range(3):
#         model = model_origin
#         model.n_classes = 5
#         model.initialize() 
#         print("the dimension of parameters is:", model_origin.logvar_c.size())
#         data = torch.load(init_model_path)
#         print("the dimension of parameters is:", model.logvar_c.size())
#         model.load_state_dict(data['model'])
#         model.to(model.device)
#         model.n_classes = 1 
#         model.initialize()   # reinit to have different seeds
#         print("the dimension of parameters is:", model.logvar_c.size())
#         model.train_model()
#         trained_models_cluster_folder = os.path.join(trained_models_folder, f"{model.n_classes}_clusters")
#         try:
#                 os.makedirs(trained_models_cluster_folder)
#         except Exception as e:
#                 print(e)
#         trained_model_path = os.path.join(trained_models_cluster_folder, f'trained_model_{i}.pt')
#         state_dict = OrderedDict((k,v) for k,v in model.state_dict().items() if not k.startswith('net.tmp_var'))
#         torch.save({'model': state_dict}, trained_model_path)
#         del model

# if args.evaluate:
#         model = model_origin
#         model.n_classes = 1 
#         save_dir = trained_models_cluster_folder
#         cca_evaluation(args, model, save_dir)
