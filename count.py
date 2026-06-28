import argparse
import json
import os
import warnings
from collections import Counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tva.dataset import HRDataset, fn_collate
from tva.tokenizers import get_tokenizer
from tva.model import BaseModel

warnings.filterwarnings('ignore', category=UserWarning)


def main(cfgs: argparse.Namespace) -> None:
    '''Main function to count token frequencies.

    Args:
        cfgs: Configurations.
    '''
    tokenizer = get_tokenizer(cfgs.tokenizer)
    tokenizer.load(os.path.join(cfgs.dir_tokenizer, f'{cfgs.idx_fold}.json'))
    model = BaseModel(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.num_channel,
        tokenizer.size,
        cfgs.len_seq,
    ).to(cfgs.device)

    ckp = torch.load(cfgs.checkpoint, map_location=torch.device(cfgs.device))
    model.load_state_dict(ckp['model'], strict=False)

    dataset_val = HRDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        tokenizer,
        model.ratio_ds,
        cfgs.idx_fold,
        cfgs.len_seq,
        cache=cfgs.cache,
    )
    dataloader_val = DataLoader(
        dataset_val,
        cfgs.size_batch,
        num_workers=cfgs.num_worker,
        collate_fn=fn_collate,
    )

    counter_token = Counter()
    model.eval()

    with torch.no_grad():
        for x, _, lens_x, _ in dataloader_val:
            x = x.to(cfgs.device)
            out = model(x)

            pred = torch.argmax(out, dim=2)

            for i, len_x in enumerate(lens_x):
                pred_token = pred[i, : len_x // model.ratio_ds]
                pred_token = torch.unique_consecutive(pred_token)
                tokens = pred_token.cpu().numpy()
                counter_token.update(tokens)

    freq_token = {
        tokenizer.decode([id_token]): count
        for id_token, count in counter_token.most_common()
        if id_token != 0
    }
    ratio_token = {
        id_token: count / sum(freq_token.values())
        for id_token, count in freq_token.items()
    }

    freq_len = Counter()

    for str_token, count in freq_token.items():
        freq_len[len(str_token)] += count

    ratio_len = {
        length: count / freq_len.total()
        for length, count in sorted(freq_len.items())
    }

    counts = list(ratio_token.values())
    mean = np.mean(counts).item()
    std = np.std(counts).item()

    with open(
        os.path.join(
            cfgs.dir_work, str(cfgs.idx_fold), 'token_frequencies.json'
        ),
        'w',
    ) as f:
        json.dump(
            {
                'ratio_token': ratio_token,
                'ratio_length': ratio_len,
                'mean': mean,
                'std': std,
            },
            f,
            ensure_ascii=False,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Count token frequencies in predictions on the validation set.'
        )
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
