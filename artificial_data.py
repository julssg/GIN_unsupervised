import argparse
import torch
import os
from collections import OrderedDict
from model import GIN

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
parser.add_argument('--init_identity', type=int, default=1,
                    help='Initialize the network as the identity (1, default) or not (0)')
args = parser.parse_args()
assert args.incompressible_flow in [0,1], 'Argument should be 0 or 1'
assert args.empirical_vars in [0,1], 'Argument should be 0 or 1'
assert args.unsupervised in [0,1], 'Argument should be 0 or 1'
assert args.init_identity in [0,1], 'Argument should be 0 or 1'

model = GIN(dataset='10d', 
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

state_dict = OrderedDict((k,v) for k,v in model.state_dict().items() if not k.startswith('net.tmp_var'))
os.makedirs(os.path.join(model.save_dir, 'model_save'))
os.makedirs(os.path.join(model.save_dir, 'figures'))
init_model_path = os.path.join(model.save_dir, 'model_save', 'init.pt')
torch.save({'model': state_dict}, init_model_path )

for i in range(5):
        data = torch.load(init_model_path)
        model.load_state_dict(data['model'])
        model.to(model.device)
        model.train_model()


