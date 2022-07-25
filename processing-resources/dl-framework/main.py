from torch.utils.data import DataLoader
import pickle as pkl
import sys

import collate_functions
import utilities
import pipeline
import process_results
import models
import torch


base_dir = sys.argv[1]
config_dir = sys.argv[2]



run_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/run_config.txt'
data_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/data_config.txt'
preprocessing_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/preprocessing_config.txt'
model_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/model_config.txt'
optimization_config_path = f'{base_dir}/cnc-task-3/script-development/{config_dir}/optimization_config.txt'

run_config_dict = utilities.generate_config_dict(run_config_path)
data_config_dict = utilities.generate_config_dict(data_config_path)
preprocessing_config_dict = utilities.generate_config_dict(preprocessing_config_path)
model_config_dict = utilities.generate_config_dict(model_config_path)
optimization_config_dict = utilities.generate_config_dict(optimization_config_path)


print(run_config_dict)
print(data_config_dict)
print(preprocessing_config_dict)
print(model_config_dict)
print(optimization_config_dict)




hparams = getattr(models, model_config_dict["model"]).generate_hyperparameter_sets(optimization_config_dict, model_config_dict)
for hparam in hparams:
    print(hparam)
run_config = utilities.run_config_from_config_dict(run_config_dict, model_config_dict, hparams)
data_config = utilities.data_config_from_config_dict(data_config_dict, run_config, base_dir)
preprocessing_config = utilities.preprocessing_config_from_config_dict(preprocessing_config_dict)


print(run_config)
print(data_config)
print(preprocessing_config)

'''

utilities.generate_output_folder_structure(run_config, data_config, preprocessing_config)
folds = utilities.generate_folds(data_config)

all_metrics = [pipeline.cross_validate(i, run_config.hparams[i], folds, run_config, data_config, preprocessing_config)
               for i in range(len(run_config.hparams))]

utilities.save_metrics(run_config, data_config, all_metrics)
'''

all_mc_samples = process_results.identify_misclassified_sample_ids_across_fpe(data_config, run_config)
all_enc_samples = process_results.identify_encountered_misclassified_samples_across_fpe(all_mc_samples)
process_results.generate_misclassified_sample_tracking_csvs(all_mc_samples, all_enc_samples, data_config)
metrics = process_results.load_metrics_from_file(run_config, data_config)
process_results.generate_metrics_tables_plts(metrics, data_config)
process_results.generate_metrics_report(run_config_dict, run_config, data_config_dict, data_config,
                                        preprocessing_config_dict)
