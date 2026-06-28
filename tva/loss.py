import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CTCLoss', 'EmbeddingCTCLoss', 'get_fn_loss']


class CTCLoss(nn.Module):
    '''Custom CTCLoss with probability smoothing.

    This module applies a smoothing factor to the input probability
    distribution before computing the standard CTC loss.

    Args:
        alpha_smooth: Smooth factor for input probability smoothing. If 0,
            original probabilities are used. Defaults to 1e-6.
        blank: Blank label index. Defaults to 0.
        reduction: Specifies the reduction to apply to the output. Options:
            'none', 'mean', 'sum'. Defaults to 'mean'.
        zero_infinity: Whether to zero infinite losses and associated
            gradients. Defaults to False.

    Attributes:
        alpha_smooth: Probability smoothing factor.
        blank: Blank label index.
        reduction: Reduction mode.
        zero_infinity: Infinite loss handling flag.
    '''

    def __init__(
        self,
        alpha_smooth: float = 1e-6,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = False,
    ) -> None:
        super().__init__()

        self.alpha_smooth = alpha_smooth
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        lens_pred: torch.Tensor,
        len_target: torch.Tensor,
    ) -> torch.Tensor:
        '''Forward method for computing smoothed CTC loss.

        Args:
            preds: Non-log probability predictions.
            targets: Target labels.
            lens_pred: Lengths of the inputs.
            len_target: Lengths of the targets.

        Returns:
            Calculated loss value.
        '''
        if self.alpha_smooth:
            preds = self.smooth_probs(preds, self.alpha_smooth)

        preds = preds.log()
        loss = nn.functional.ctc_loss(
            preds,
            targets,
            lens_pred,
            len_target,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )

        return loss

    @staticmethod
    def smooth_probs(probs: torch.Tensor, alpha: float = 1e-6) -> torch.Tensor:
        '''Smooths a probability distribution with a uniform distribution.

        Args:
            probs: Original probability distribution.
            alpha: Smoothing factor. Defaults to 1e-6.

        Returns:
            Smoothed probability distribution.
        '''
        num_cls = probs.shape[-1]
        distr_uni = torch.full_like(probs, 1.0 / num_cls)
        probs = (1 - alpha) * probs + alpha * distr_uni

        # ensure the smoothed probabilities sum to 1
        probs /= probs.sum(dim=-1, keepdim=True)

        return probs
