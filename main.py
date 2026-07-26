import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from tva.dataset import HRDataset, fn_collate
from tva.decoder_ctc import BestPath
from tva.evaluate import evaluate
from tva.loss import CTCLoss
from tva.tokenizers import get_tokenizer
from tva.manager import RunManager
from tva.model import BaseModel
from tva.utils import seed_everything, seed_worker
from tva.visualize import visualize

warnings.filterwarnings('ignore', category=UserWarning)


def train_one_epoch(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.SequentialLR,
    manager: RunManager,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader: Dataloader of training set.
        model: Model instance.
        fn_loss: Loss function module.
        optimizer: Optimizer instance.
        scaler: Scaler for mixed-precision training.
        lr_scheduler: Learning rate scheduler.
        manager: Running manager instance.
        epoch: Current epoch number.
    '''
    manager.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(manager.cfgs.device), y.to(manager.cfgs.device)

        optimizer.zero_grad()

        with torch.autocast('cuda', torch.float16):
            out = model(x)
            loss = fn_loss(
                out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        manager.update_iteration(
            idx,
            loss.item(),
            lr_scheduler.get_last_lr()[0],
        )

    manager.summarize_epoch()

    # save checkpoints every freq_save epoch
    if manager.check_step(epoch + 1, 'save'):
        manager.save_checkpoint(model.state_dict())


def test(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    manager: RunManager,
    ctc_decoder: BestPath,
    epoch: int | None = None,
) -> None:
    '''Test the model.

    Args:
        dataloader: DataLoader of test set.
        model: Model instance.
        fn_loss: Loss function module.
        manager: Running manager instance.
        ctc_decoder: CTC decoder instance.
        epoch: Epoch number. Defaults to None.
    '''
    preds, labels = [], []
    manager.initialize_epoch(epoch, len(dataloader), True)
    model.eval()

    with torch.no_grad():
        for idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x, y = x.to(manager.cfgs.device), y.to(manager.cfgs.device)

            out = model(x)
            loss = fn_loss(
                out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
            )

            manager.update_iteration(idx, loss.item())

            # decode and cache results every freq_eval epoch
            if manager.check_step(epoch + 1, 'eval'):
                for pred, len_pred, label in zip(
                    out.cpu(), len_x // model.ratio_ds, y.cpu()
                ):
                    preds.append(ctc_decoder.decode(pred[:len_pred]))
                    labels.append(ctc_decoder.decode(label, True))

    manager.summarize_epoch()

    # evaluate every freq_eval epoch
    if manager.check_step(epoch + 1, 'eval'):
        visualize(
            preds, labels, manager.cfgs.categories[1:], manager.dir_vis, epoch
        )
        results_eval = evaluate(preds, labels)
        manager.update_evaluation(results_eval, preds, labels)


def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs: Configurations.
    '''
    # initialize the environment
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    tokenizer = get_tokenizer(cfgs.tokenizer)
    tokenizer.load(os.path.join(cfgs.dir_tokenizer, f'{cfgs.idx_fold}.json'))
    ctc_decoder = BestPath(tokenizer)
    model = BaseModel(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.num_channel,
        tokenizer.size,
        cfgs.len_seq,
    ).to(cfgs.device)
    dataset_test = HRDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        tokenizer,
        model.ratio_ds,
        cfgs.idx_fold,
        cfgs.len_seq,
        cache=cfgs.cache,
    )
    dataloader_test = DataLoader(
        dataset_test,
        cfgs.size_batch,
        num_workers=cfgs.num_worker,
        collate_fn=fn_collate,
    )
    fn_loss = CTCLoss()
    epoch_start = 0

    if not cfgs.test:
        dataset_train = HRDataset(
            os.path.join(cfgs.dir_dataset, 'train.json'),
            tokenizer,
            model.ratio_ds,
            cfgs.idx_fold,
            cfgs.len_seq,
            cfgs.aug,
            cfgs.cache,
            cfgs.num_concat,
        )
        dataloader_train = DataLoader(
            dataset_train,
            cfgs.size_batch,
            True,
            num_workers=cfgs.num_worker,
            collate_fn=fn_collate,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )
        optimizer = torch.optim.AdamW(model.parameters(), cfgs.lr)
        scaler = GradScaler()
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(
                    optimizer,
                    0.01,
                    total_iters=len(dataloader_train) * cfgs.epoch_warmup,
                ),
                CosineAnnealingLR(
                    optimizer,
                    len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup),
                ),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )

    # load checkpoint for testing if given
    if cfgs.checkpoint:
        ckp = torch.load(cfgs.checkpoint, weights_only=False)
        model.load_state_dict(ckp['model'], strict=False)
        manager.log(f'Load checkpoint from {cfgs.checkpoint}')

    # start running
    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                -1,
            )
            break
        else:
            train_one_epoch(
                dataloader_train,
                model,
                fn_loss,
                optimizer,
                scaler,
                lr_scheduler,
                manager,
                e,
            )
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                e,
            )

    if not cfgs.test:
        manager.summarize_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()
    # args.config = 'configs/train.yaml'  # ONLY for debugging

    with open(args.config, 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
