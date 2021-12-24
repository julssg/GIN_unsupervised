import argparse
import os
from time import time
import torch
from model import GIN
from collections import OrderedDict
# from evaluate import cca_evaluation, evaluate_stability, evaluate_stability_many_data
from evaluate import mcc_evaluation
from plot import plot_mcc_emnist



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
                        help='Number of runs (default 1).')
        parser.add_argument('--init', type=str, default='xavier',
                        help='Initialization method, can be chosen to be "batch", "supervised", "supervised_pretraining" or "xavier" uniform (default). Batch and supervised \
                                initialization include also a re-initialization after the 1, 2 and 5th epoch.')
        parser.add_argument('--evaluate', type=int, default=0,
                    help='State whether model should be evaluated (1) or not (0, default)')
                                

        args = parser.parse_args()

        assert args.incompressible_flow in [0,1], 'Argument should be 0 or 1'
        assert args.empirical_vars in [0,1], 'Argument should be 0 or 1'
        assert args.unsupervised in [0,1], 'Argument should be 0 or 1'
        if args.unsupervised == 1:
                assert args.init in ["batch", "supervised", "xavier", "supervised_pretraining"], \
                        'init methods only if training unsupervised, should be in ["batch", "supervised", "xavier", "supervised_pretraining"]'
        else:
                assert args.init, \
                        'init methods only if training unsupervised, should be in ["batch", "supervised", "xavier", "supervised_pretraining"]'
        assert args.evaluate in [0,1], 'Argument should be 0 or 1'

        return args


def main():
        args = arg_parse()

        model = model_init(args)

        if args.n_runs == 1:
                model.train_model()
        else:
                # timestamp = str(int(time()))
                # save_dir = os.path.join('./emnist_save/', 'many_runs', timestamp)
                # os.makedirs(save_dir)

                # for run in range(args.n_runs):
                #         print(f"Starting {run+1} run:")
                #         model.train_model()
                #         save(model, os.path.join(save_dir, f'{run+1}.pt'))

                #         model = model_init(args)

                if args.evaluate:
                        save_dir = os.path.join('./emnist_save/', 'many_runs', "1630500545")
                        # cca_evaluation(args, model_init(args), save_dir )
                        # evaluate_stability(args, model_init(args), save_dir, cross_validation=True)
                        df = mcc_evaluation(model_init(args), args, save_dir, cross_validation=False)
                        plot_mcc_emnist(df, save_dir)
                        # evaluate_stability_many_data(args, model_init(args), save_dir, cross_validation=True)

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
                        n_classes=args.n_clusters,
                        init_method=args.init)
        return model


if __name__ == '__main__':
    main()
