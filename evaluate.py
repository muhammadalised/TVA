import argparse
import json
import os
from glob import glob
from typing import Any

import numpy as np
import torch
import yaml
from thop import profile

from tva.model import BaseModel
from tva.tokenizers import get_tokenizer


def get_mean_std_cv(
    cfgs: dict[str, Any], results: dict[str, Any] | None = None
) -> dict[str, Any]:
    '''Calculates statistics for cross-validation results.

    Iterates through result JSON files in the work directory, extracts the
    best Character Error Rate (CER) and Word Error Rate (WER), and computes
    mean and standard deviation.

    Args:
        cfgs: Configuration dictionary containing 'dir_work' and 'test' keys.
        results: Dictionary to append results to. Defaults to None.

    Returns:
        Updated dictionary containing 'cer' and 'wer' statistics (raw, mean,
        std).
    '''
    if results is None:
        results = {}

    cer, wer = {}, {}

    if paths_result := glob(
        os.path.join(
            cfgs['dir_work'],
            '*',
            'test_20*.json' if cfgs['test'] else 'train_20*.json',
        )
    ):
        for i, path_result in enumerate(sorted(paths_result)):
            with open(path_result, 'r') as f:
                result_fd = json.load(f)

            if cfgs['test']:
                result_best = result_fd['-1']['evaluation']
            else:
                epoch_best = result_fd['best']['character_error_rate'][0]
                result_best = result_fd[str(epoch_best)]['evaluation']

            cer[str(i)] = result_best['character_error_rate']
            wer[str(i)] = result_best['word_error_rate']

        results['cer'] = {
            'raw': cer,
            'mean': np.mean(list(cer.values())).item(),
            'std': np.std(list(cer.values())).item(),
        }
        results['wer'] = {
            'raw': wer,
            'mean': np.mean(list(wer.values())).item(),
            'std': np.std(list(wer.values())).item(),
        }
        results = {k: v for k, v in sorted(results.items())}

    return results


def get_macs_params(
    cfgs: dict[str, Any], results: dict[str, Any] | None = None
) -> dict[str, Any]:
    '''Calculates the computational cost and model size.

    Computes the number of parameters and Multiply-Accumulate operations
    (MACs) using a dummy input.

    Args:
        cfgs: Configuration dictionary containing architecture details and
            channel counts. The dummy sequence length is scaled by one plus
            'num_concat'.
        results: Dictionary to append results to. If None, a new dictionary
            is created. Defaults to None.

    Returns:
        Updated dictionary containing 'macs' and 'params'.
    '''
    if results is None:
        results = {}

    tokenizer = get_tokenizer(cfgs['tokenizer'])
    tokenizer.load(
        os.path.join(cfgs['dir_tokenizer'], f'{cfgs["idx_fold"]}.json')
    )
    model = BaseModel(
        cfgs['arch_en'],
        cfgs['arch_de'],
        cfgs['num_channel'],
        # tokenizer.size,
        500,
        cfgs['len_seq'],
    ).eval()

    # match HRDataset: original sample plus num_concat extra samples
    num_samples = int(cfgs['num_concat']) + 1
    len_seq = 1024 * num_samples
    x = torch.randn(1, cfgs['num_channel'], len_seq)
    macs, params = profile(model, inputs=(x,))

    results['macs'] = int(macs)
    results['params'] = int(params)
    results = {k: v for k, v in sorted(results.items())}

    return results


def main(path_cfg: str) -> None:
    '''Main execution routine for single-experiment evaluation.

    Loads configuration, creates work directories, calculates cross-validation
    statistics, computes model complexity, and saves the aggregated results to
    JSON.

    Args:
        path_cfg: Path to the configuration YAML file.
    '''
    with open(path_cfg, 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)

    os.makedirs(cfgs['dir_work'], exist_ok=True)

    path_results = os.path.join(cfgs['dir_work'], 'results.json')

    if os.path.isfile(path_results):
        with open(path_results, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    results = get_mean_std_cv(cfgs, results)
    results = get_macs_params(cfgs, results)

    with open(path_results, 'w') as f:
        json.dump(results, f)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()

    if os.path.isfile(args.config):
        main(args.config)
