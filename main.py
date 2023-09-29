"""
Main script for running LGBM/ Type2 GNN training/ bayesian optimization tuning.
"""
import sys
# pyg_path = '/home/disco-computing/begressy/IBM/PYG/pytorch_geometric'
pyg_path = '/home/disco-computing/begressy/IBM/Pattern-GNN/pytorch_geometric'
sys.path.insert(0, pyg_path)

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import logging
from train_helpers.train_gnn_type2 import bayes_opt_tune_type2_gnn, train_gnn_type2, test_only_gnn_type2
from train_helpers.train_setup import get_eth_data, get_aml_data, get_dir_dataset
from train_helpers.simulator import get_gnn_data_from_simulator
from utils.util import get_settings, set_seed
import time
import torch


def main():
    """
    Runs data loading, preprocessing, training, and evaluation scripts
    :return: None
    """
    config, model_settings, args, log_dir, perfromance_csv = get_settings()

    if config.simulator == "eth":
        tr_data, val_data, te_data, val_folds, te_inds = get_eth_data(config, args)
        data_list = [tr_data, val_data, te_data]
    elif config.simulator == "aml-e":
        tr_data, val_data, te_data, val_folds, te_inds = get_aml_data(config, args)
        data_list = [tr_data, val_data, te_data]
    elif config.simulator == "sim":
        tr_data, val_data, te_data = get_gnn_data_from_simulator(config, args, max_time=10)
        data_list = [tr_data, val_data, te_data]  
        val_folds = [(None, None)]
        te_inds = None
    elif config.simulator == "dir":
        tr_data, val_data, te_data, val_folds, te_inds = get_dir_dataset(
            name=config.edge_file, root_dir=config.data_dir, homophily=None, undirected=False, self_loops=False, transpose=False, seeds=args.num_seeds)
        data_list = [tr_data, val_data, te_data]
    else:
        raise ValueError('did you forget to set config.simulator?')

    logging.debug(f"Start: adding ports [{args.ports}] and time deltas [{args.time_deltas}]")
    for data in data_list:
        logging.info(f'y_sum, y_len = {sum(data.y)}, {len(data.y)}')
        if args.ports:
            if args.random_ports:
                logging.info(f"Using random port ordering")
            data.add_ports(random=args.random_ports)
        if args.add_node_ids:
            data.add_nodeIDs()
        if args.time_deltas:
            data.add_time_deltas()
        if config.simulator == "aml-e":
            data.add_edgeIDs()
    logging.debug(f"Done:  adding ports [{args.ports}] and time deltas [{args.time_deltas}]")
    # Run with Bayesian optimization
    if config.tune:
        if config.simulator == "dir": args.torch_seed = 0
        params = model_settings.bayes_opt_params
        logging.debug(f"len(val_folds) = {len(val_folds)}")
        bayes_opt_tune_type2_gnn(
            tr_data, te_data, val_data, val_folds, te_inds, config, model_settings, args, perfromance_csv, log_dir
            )
    # Run inference only with saved model
    elif args.inference:
        params = model_settings.params
        test_only_gnn_type2(
            tr_data, te_data, val_data, val_folds, te_inds, config, model_settings, args, perfromance_csv, log_dir, 
            save_run_embed=config.generate_embedding, tb_logging=args.tb, **params
            )
    # Run multiple validation folds for dirGNN datasets
    elif config.simulator == "dir":
        logging.info(f"Going through {len(val_folds)} validation folds.")
        params = model_settings.params
        logging.debug(f"len(val_folds) = {len(val_folds)}")
        for val_fold in val_folds:
            # val_fold, te_inds = fold
            te_inds = ~(val_fold[0] + val_fold[1])
            args.torch_seed = 0
            fold_start_time = time.time()
            train_gnn_type2(
                tr_data, te_data, val_data, [val_fold], te_inds, config, model_settings, args, perfromance_csv, log_dir, 
                save_run_embed=config.generate_embedding, tb_logging=args.tb, **params
                )
            logging.info(f"Fold Done. Total Training Time [TTT] = {time.time() - fold_start_time}")

    # Run several seeds for getting mean and standard deviations of the metrics
    elif args.num_seeds > 1:
        params = model_settings.params
        for i in range(args.num_seeds):
            args.torch_seed = i
            seed_start_time = time.time()
            train_gnn_type2(
                tr_data, te_data, val_data, val_folds, te_inds, config, model_settings, args, perfromance_csv, log_dir, 
                save_run_embed=config.generate_embedding, tb_logging=args.tb, **params
                )
            logging.info(f"Seed Done. Total Training Time [TTT] = {time.time() - seed_start_time}")
    # Single run
    else:
        params = model_settings.params
        train_gnn_type2(
            tr_data, te_data, val_data, val_folds, te_inds, config, model_settings, args, perfromance_csv, log_dir, 
            save_run_embed=config.generate_embedding, tb_logging=args.tb, **params
            )

    return None
    

if __name__ == "__main__":
    main()
    logging.info('DONE')
