import argparse
import os
from time import time
import torch
from model import GIN
from collections import OrderedDict
from evaluate import mcc_evaluation


def arg_parse():
        parser = argparse.ArgumentParser(description='Experiments on EMNIST with GIN (training script)')
        parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs (default 100)')
        parser.add_argument('--epochs_per_line', type=int, default=1,
                        help='Print a new line after this many epochs (default 1)')
        parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learn rate (default 3e-4)')
        parser.add_argument('--lr_schedule', nargs='+', type=int, default=[50],
                        help='Learn rate schedule (decrease lr by factor of 10 at these epochs, default [50]). \
                                Usage example: --lr_schedule 20 40')
        parser.add_argument('--batch_size', type=int, default=240,
                        help='Batch size (default 240)')
        parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save a new checkpoint and make plots after this many epochs (default 10)')
        parser.add_argument('--data_root_dir', type=str, default='./',
                        help='Directory in which \'EMNIST\' directory storing data is located (defaults to current directory). If the data is not found here you will be prompted to download it')
        parser.add_argument('--incompressible_flow', type=int, default=1,
                        help='Use an incompressible flow (GIN) (1, default) or compressible flow (GLOW) (0)')
        parser.add_argument('--empirical_vars', type=int, default=1,
                        help='Estimate empirical variables (means and stds) for each batch (1, default) or learn them along \
                                with model weights (0)')
        parser.add_argument('--unsupervised', type=int, default=0,
                        help='State whether model should be trained supervised (0, default) or unsupervised \
                                with leared clustering in latent space (1)')
        parser.add_argument('--n_clusters', type=int, default=40,
                        help='Number of components in gaussian mixture (default 40)')
        parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs (default 1), if 0 selected, than only evaluation will be performed \
                                path needs to be given in emnist.py script.')

        args = parser.parse_args()

        assert args.incompressible_flow in [0,1], 'Argument should be 0 or 1'
        assert args.empirical_vars in [0,1], 'Argument should be 0 or 1'
        assert args.unsupervised in [0,1], 'Argument should be 0 or 1'

        return args


def main():
        args = arg_parse()

        model = model_init(args)

        if args.n_runs == 1:
                model.train_model()
        elif args.n_runs == 0:
                save_dir = os.path.join('./emnist_save/', 'many_runs', '1630500545')
                mcc_evaluation(args, model, save_dir)
        else:
                timestamp = str(int(time()))
                save_dir = os.path.join('./emnist_save/', 'many_runs', timestamp)
                os.makedirs(save_dir)

                for run in range(args.n_runs):
                        print(f"Starting {run+1} run:")
                        model.train_model()
                        save(model, os.path.join(save_dir, f'{run+1}.pt'))

                        model = model_init(args)

                # mcc_evaluation(args, model, save_dir)


def save(model, fname):
        state_dict = OrderedDict((k,v) for k,v in model.state_dict().items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, fname)


def model_init(args):
        model = GIN(dataset='EMNIST', 
                        n_epochs=args.n_epochs, 
                        epochs_per_line=args.epochs_per_line, 
                        lr=args.lr, 
                        lr_schedule=args.lr_schedule, 
                        batch_size=args.batch_size, 
                        save_frequency=args.save_frequency, 
                        data_root_dir=args.data_root_dir, 
                        incompressible_flow=args.incompressible_flow, 
                        empirical_vars=args.empirical_vars,
                        unsupervised=args.unsupervised,
                        n_classes=args.n_clusters)
        return model


if __name__ == '__main__':
    main()