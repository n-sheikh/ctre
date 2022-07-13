from torch.utils.data import DataLoader
import pickle as pkl
import sys

import collate_functions
import utilities
import pipeline
import process_results
import models

base_dir = sys.argv[1]
config_dir = sys.argv[2]


run_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/run_config.txt'
data_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/data_config.txt'
preprocessing_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/preprocessing_config.txt'

run_config_dict = utilities.generate_config_dict(run_config_path)
data_config_dict = utilities.generate_config_dict(data_config_path)
preprocessing_config_dict = utilities.generate_config_dict(preprocessing_config_path)
hparams = utilities.generate_hyperparameter_sets(run_config_dict)
run_config = utilities.run_config_from_config_dict(run_config_dict, hparams)
data_config = utilities.data_config_from_config_dict(data_config_dict, run_config, base_dir)
preprocessing_config = utilities.preprocessing_config_from_config_dict(preprocessing_config_dict)
utilities.generate_output_folder_structure(run_config, data_config, preprocessing_config)
folds = utilities.generate_folds(data_config)
all_metrics = [pipeline.cross_validate(i, run_config.hparams[i], folds, run_config, data_config, preprocessing_config)
               for i in range(len(run_config.hparams))]
utilities.save_metrics(run_config, data_config, all_metrics)
all_mc_samples = process_results.identify_misclassified_sample_ids_across_fpe(data_config, run_config)
all_enc_samples = process_results.identify_encountered_misclassified_samples_across_fpe(all_mc_samples)
process_results.generate_misclassified_sample_tracking_csvs(all_mc_samples, all_enc_samples, data_config)
metrics = process_results.load_metrics_from_file(run_config, data_config)
process_results.generate_metrics_tables_plts(metrics, data_config)
process_results.generate_metrics_report(run_config_dict, run_config, data_config_dict, data_config,
                                        preprocessing_config_dict)

