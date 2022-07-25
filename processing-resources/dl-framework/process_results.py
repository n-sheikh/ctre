import csv
import os
import pickle as pkl
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date


def load_misclassified_samples_from_file(path):
    false_positives = []
    false_negatives = []
    with open(path) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if row[1] == '0' and row[2] == '1':
                false_positives.append(row[0])
            elif row[1] == '1' and row[2] == '0':
                false_negatives.append(row[0])
        return false_positives, false_negatives


def identify_misclassified_sample_ids_across_fpe(data_config, run_config):
    all_misclassified_samples = {}
    output_folder_path = data_config.output_folder_path
    for i in range(data_config.nos_folds):
        all_misclassified_samples[f'fold_{i}'] = {}
        for j in range(len(run_config.hparams)):
            all_misclassified_samples[f'fold_{i}'][f'hparam_config_id_{j}'] = []
            for k in range(run_config.hparams[j]['max_epochs']):
                path = output_folder_path + f'/{j}/results/classified_samples/fold_{i}_epoch_{k}.csv'
                all_misclassified_samples[f'fold_{i}'][f'hparam_config_id_{j}'].append(
                    load_misclassified_samples_from_file(path))
    return all_misclassified_samples


def identify_encountered_misclassified_samples_in_fold_across_pe(fold_misclassified_samples):
    false_positives = []
    false_negatives = []
    for key in fold_misclassified_samples:
        for idx in range(len(fold_misclassified_samples[key])):
            false_positives = false_positives + fold_misclassified_samples[key][idx][0]
            false_negatives = false_negatives + fold_misclassified_samples[key][idx][1]
    return set(false_positives), set(false_negatives)


def identify_encountered_misclassified_samples_across_fpe(all_misclassified_samples):
    encountered_samples = {}
    for key, value in all_misclassified_samples.items():
        val = identify_encountered_misclassified_samples_in_fold_across_pe(value)
        encountered_samples[key] = val
    return encountered_samples


def construct_misclassified_matrix(fold_misclassified_samples, fold_encountered_samples):
    matrix = {"false_positives": [], "false_negatives": []}
    encountered_false_positives = fold_encountered_samples[0]
    encountered_false_negatives = fold_encountered_samples[1]
    for key in fold_misclassified_samples.keys():
        false_positives_across_e = [ep_misclassified_samples[0] for ep_misclassified_samples in fold_misclassified_samples[key]]
        false_negatives_across_e = [ep_misclassified_samples[1] for ep_misclassified_samples in fold_misclassified_samples[key]]
        track_false_positives_across_e = []
        track_false_negatives_across_e = []
        for fp in encountered_false_positives:
            track_false_positives_across_e.append([1 if fp in fp_in_e else 0 for fp_in_e in false_positives_across_e])
        for fn in encountered_false_negatives:
            track_false_negatives_across_e.append([1 if fn in fn_in_e else 0 for fn_in_e in false_negatives_across_e])
        matrix["false_positives"].append(track_false_positives_across_e)
        matrix["false_negatives"].append(track_false_negatives_across_e)
    return encountered_false_positives, matrix["false_positives"], encountered_false_negatives, matrix["false_negatives"]


def generate_misclassified_matrix_csv_ds(encountered, matrix):
    csv_ds = []
    header = [sample_id for sample_id in encountered]
    csv_ds.append(itertools.chain(header))
    for row in matrix:
        csv_ds.append(itertools.chain(row))
    return csv_ds


def generate_misclassified_sample_tracking_csvs(all_misclassified_samples, all_encountered_samples, data_config):
    with open(data_config.output_folder_path + f'results_report/misclassified_samples_tracking/all_misclassified_samples.pkl', 'wb') as f:
        pkl.dump(all_misclassified_samples, f)
    with open(data_config.output_folder_path + f'results_report/misclassified_samples_tracking/all_encountered_samples.pkl', 'wb') as f:
        pkl.dump(all_encountered_samples, f)
    for key in all_misclassified_samples.keys():
        enc_false_positives, fp_matrix, enc_false_negatives, fn_matrix = construct_misclassified_matrix(
            all_misclassified_samples[key], all_encountered_samples[key])
        csv_ds_false_positives = generate_misclassified_matrix_csv_ds(enc_false_positives, fp_matrix)
        csv_ds_false_negatives = generate_misclassified_matrix_csv_ds(enc_false_negatives, fn_matrix)
        with open(data_config.output_folder_path +
                  f'results_report/misclassified_samples_tracking/{key}/false_positives.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(csv_ds_false_positives)
        with open(data_config.output_folder_path +
                  f'results_report/misclassified_samples_tracking/{key}/false_negatives.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(csv_ds_false_negatives)


def load_metrics_from_file(run_config, data_config):
    metric_types = ['train_loss', 'test_loss', 'accuracy_score', 'f1_score', 'precision_score', 'matthews_corrcoef',
                    'recall_score']
    metrics = {}
    for i in range(data_config.nos_folds):
        metrics[f'fold_{i}'] = {}
        for mt in metric_types:
            metrics[f'fold_{i}'][mt] = []
            for j in range(len(run_config.hparams)):
                with open(f'{data_config.output_folder_path}{j}/results/{mt}.pkl', 'rb') as f:
                    hparam_metrics = pkl.load(f)
                    metrics[f'fold_{i}'][mt].append(hparam_metrics[i])
    return metrics


def generate_fold_dfs(fold_metrics):
    dfs = {}
    nos_epochs = len(fold_metrics[list(fold_metrics.keys())[0]][0])
    nos_hparam_settings = len(fold_metrics[list(fold_metrics.keys())[0]])
    for metric in fold_metrics:
        np_metrics = np.array(fold_metrics[metric]).round(3)
        dfs[metric] = pd.DataFrame(data=np_metrics, columns=[f'ep {i}' for i in range(nos_epochs)],
                                   index=[f'hp_{i}' for i in range(nos_hparam_settings)])
    return dfs


def generate_fold_plts(fold_metrics, fold_id, data_config):
    nos_epochs = len(fold_metrics[list(fold_metrics.keys())[0]][0])
    for metric in fold_metrics:
        plt.clf()
        for i in range(len(fold_metrics[metric])):
            plt.plot([f"ep {i}" for i in range(nos_epochs)], fold_metrics[metric][i], label=f'hp_{i}')
        plt.legend()
        plt.title(f"{metric} across epochs for {fold_id} across epochs")
        plt.ylabel(f'{metric}')
        plt.xlabel('Epochs')
        path = data_config.output_folder_path + f'results_report/metrics/{fold_id}/{metric}.jpg'
        plt.savefig(path)


def generate_metrics_tables_plts(metrics, data_config):
    with open(data_config.output_folder_path + f'results_report/metrics/all_metrics.pkl', 'wb') as f:
        pkl.dump(metrics, f)
    for fold_id in metrics.keys():
        generate_fold_plts(metrics[fold_id], fold_id, data_config)
        dfs = generate_fold_dfs(metrics[fold_id])
        for metric in dfs.keys():
            with open(data_config.output_folder_path + f'results_report/metrics/{fold_id}/{metric}.tex', 'w') as f:
                f.write(dfs[metric].to_latex())


def generate_graphics_path(data_config):
    graphics_path = ''
    for i in range(data_config.nos_folds):
       graphics_path = graphics_path + f"\\graphicspath{{ {{./metrics/fold_{i}/}} }}\n"
    return graphics_path


def generate_header(run_config):
    header = f"\\title{{Results: {run_config.run_identifier}}}\n" \
             f"\\author{{Nadia Sheikh}}\n" \
             f"\\date{{{date.today()}}}\n" \
             f"\\begin{{document}}\n" \
             f"\\maketitle\n"
    return header


def generate_configs(run_config_dict, data_config_dict, preprocessing_config_dict):
    configs = '\\section{Configurations}\n'
    configs = configs + '\\subsection{Run Configuration}\n'
    configs = configs + pd.DataFrame.from_dict(run_config_dict, orient='index').to_latex() + '\n'
    configs = configs + '\\subsection{Data Configuration}\n'
    configs = configs + pd.DataFrame.from_dict(data_config_dict, orient='index').to_latex() + '\n'
    configs = configs + '\\subsection{Preprocessing Configuration}\n'
    configs = configs + pd.DataFrame.from_dict(preprocessing_config_dict, orient='index').to_latex() + '\n'
    return configs


def generate_hyperparameter_description(hparams):
    hparams_description = '\\section{Hyperparameters}\n'
    for i in range(len(hparams)):
        hparams_description = hparams_description + f'\\subsection{{hparam\_config\_id\_{i}}}\n'
        hparams_description = hparams_description + pd.DataFrame.from_dict(hparams[i], orient='index').to_latex() + '\n'
    return hparams_description


def generate_fold_results(fold_id, data_config):
    metric_types = ['train_loss', 'test_loss', 'accuracy_score', 'f1_score', 'precision_score', 'matthews_corrcoef',
                    'recall_score']
    section_title = fold_id.replace("_", "\_")
    fold_results = f'\\section{{{section_title}}}\n'
    for metric in metric_types:
        subsection_title = metric.replace("_", "\_")
        fold_results = fold_results + f'\\subsection{{{subsection_title}}}\n'
        with open(data_config.output_folder_path + f'results_report/metrics/{fold_id}/{metric}.tex', 'r') as f:
            fold_results = fold_results + f.read() + '\n'
        fold_results = fold_results + f'\\begin{{figure}}[H]\n' \
                                      f'\\includegraphics[scale = 0.75]{{{fold_id}/{metric}}}\n' \
                                      f'\\end{{figure}}\n'
    return fold_results


def generate_metrics_report(run_config_dict, run_config, data_config_dict, data_config,
                            preprocessing_config_dict):
    doc_string = "\\documentclass{article}\n" \
                 "\\usepackage[utf8]{inputenc}\n" \
                 "\\usepackage{graphicx}\n" \
                 "\\usepackage{booktabs}\n" \
                 "\\usepackage{float}\n"
    doc_string = doc_string + generate_graphics_path(data_config)
    doc_string = doc_string + generate_header(run_config)
    doc_string = doc_string + generate_configs(run_config_dict, data_config_dict, preprocessing_config_dict)
    doc_string = doc_string + generate_hyperparameter_description(run_config.hparams)
    for i in range(data_config.nos_folds):
        doc_string = doc_string + generate_fold_results(f'fold_{i}', data_config)
    doc_string = doc_string + '\\end{document}\n'
    with open(data_config.output_folder_path + f'results_report/metrics/metrics_report.tex', 'w') as f:
        f.write(doc_string)

