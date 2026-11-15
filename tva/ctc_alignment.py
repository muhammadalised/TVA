from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = ['CTCAlignment', 'TokenAlignment', 'ctc_viterbi_align']


@dataclass(frozen=True)
class TokenAlignment:
    '''Location of one target token on the model-output timeline.

    `end_frame` is exclusive, following normal Python slicing. For example,
    `start_frame=3` and `end_frame=6` means that frames 3, 4, and 5 belong to
    this token.
    '''

    target_index: int
    token_id: int
    start_frame: int
    end_frame: int

    @property
    def num_frames(self) -> int:
        '''Number of model-output frames assigned to this token.'''
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class CTCAlignment:
    '''Best CTC path that is constrained to a known target sequence.'''

    log_score: float
    expanded_target: list[int]
    state_path: list[int]
    tokens: list[TokenAlignment]

    @property
    def token_path(self) -> list[int]:
        '''Token ID selected at every model-output frame.'''
        return [self.expanded_target[state] for state in self.state_path]


def _expand_target(target: list[int], blank_id: int) -> list[int]:
    '''Insert a CTC blank before, after, and between all target tokens.'''
    expanded = [blank_id]

    for token_id in target:
        expanded.extend([token_id, blank_id])

    return expanded


def _minimum_required_frames(target: list[int]) -> int:
    '''Return the shortest valid CTC path length for this target.'''
    repeated_neighbours = sum(
        left == right for left, right in zip(target, target[1:])
    )
    return len(target) + repeated_neighbours


def _validate_inputs(
    probabilities: torch.Tensor,
    target: Sequence[int] | torch.Tensor,
    blank_id: int,
) -> tuple[torch.Tensor, list[int]]:
    '''Validate inputs and move the small alignment problem to the CPU.'''
    if probabilities.ndim != 2:
        raise ValueError(
            'probabilities must have shape (num_frames, num_classes).'
        )
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError('probabilities must not be empty.')
    if not torch.is_floating_point(probabilities):
        raise ValueError('probabilities must be a floating-point tensor.')

    probabilities_cpu = probabilities.detach().to(
        device='cpu', dtype=torch.float64
    )
    if not torch.isfinite(probabilities_cpu).all():
        raise ValueError('probabilities must contain only finite values.')
    if (probabilities_cpu < 0).any():
        raise ValueError('probabilities cannot contain negative values.')

    row_sums = probabilities_cpu.sum(dim=1, keepdim=True)
    if (row_sums <= 0).any():
        raise ValueError('every frame must have a positive probability sum.')

    # A model Softmax already sums to one. Normalizing again only protects us
    # from small rounding differences and does not change the best path.
    probabilities_cpu = probabilities_cpu / row_sums

    if isinstance(target, torch.Tensor):
        if target.ndim != 1:
            raise ValueError('target must be a one-dimensional sequence.')
        target_ids = [int(value) for value in target.detach().cpu().tolist()]
    else:
        target_ids = [int(value) for value in target]

    if not target_ids:
        raise ValueError('target must contain at least one token.')

    num_classes = probabilities_cpu.shape[1]
    if not 0 <= blank_id < num_classes:
        raise ValueError('blank_id is outside the model vocabulary.')
    if blank_id in target_ids:
        raise ValueError('target must not contain the CTC blank token.')
    if any(token_id < 0 or token_id >= num_classes for token_id in target_ids):
        raise ValueError('target contains a token outside the model vocabulary.')

    min_frames = _minimum_required_frames(target_ids)
    if probabilities_cpu.shape[0] < min_frames:
        raise ValueError(
            f'No valid CTC path: this target needs at least {min_frames} '
            f'frames, but only {probabilities_cpu.shape[0]} were provided.'
        )

    return probabilities_cpu, target_ids


def ctc_viterbi_align(
    probabilities: torch.Tensor,
    target: Sequence[int] | torch.Tensor,
    blank_id: int = 0,
) -> CTCAlignment:
    '''Find the most likely CTC alignment for a known target sequence.

    Args:
        probabilities: Model probabilities for one sample, with shape
            `(num_frames, num_classes)`.
        target: Known target token IDs without CTC blanks.
        blank_id: Vocabulary ID of the CTC blank token. Defaults to 0.

    Returns:
        The best path and one half-open frame interval `[start, end)` for each
        target token.

    Notes:
        The returned positions refer to model-output frames, not raw IMU
        samples. TVA's BLConv encoder reduces the timeline by a factor of 8;
        converting to raw-signal positions is intentionally handled later.
    '''
    probabilities, target_ids = _validate_inputs(
        probabilities, target, blank_id
    )
    log_probabilities = probabilities.clamp_min(1e-12).log()

    expanded_target = _expand_target(target_ids, blank_id)
    num_frames = probabilities.shape[0]
    num_states = len(expanded_target)

    # scores[t, s] is the best log-probability after reaching expanded-target
    # state s at model frame t. Backpointers remember how we reached it.
    scores = torch.full(
        (num_frames, num_states),
        -torch.inf,
        dtype=torch.float64,
    )
    backpointers = torch.full(
        (num_frames, num_states),
        -1,
        dtype=torch.int64,
    )

    # At frame zero, a valid path can start at either the first blank or the
    # first target token. It cannot start any further into the target.
    scores[0, 0] = log_probabilities[0, blank_id]
    scores[0, 1] = log_probabilities[0, expanded_target[1]]

    for frame in range(1, num_frames):
        for state in range(num_states):
            current_token = expanded_target[state]

            # Option 1: stay at the same CTC state.
            best_previous_state = state
            best_previous_score = scores[frame - 1, state]

            # Option 2: move forward by one state.
            if state > 0:
                previous_score = scores[frame - 1, state - 1]
                if previous_score > best_previous_score:
                    best_previous_state = state - 1
                    best_previous_score = previous_score

            # Option 3: skip over a blank. We cannot use this transition for
            # a blank or between identical tokens; repeated tokens require a
            # real blank between them to remain separate after CTC collapse.
            can_skip_blank = (
                state > 1
                and current_token != blank_id
                and current_token != expanded_target[state - 2]
            )
            if can_skip_blank:
                previous_score = scores[frame - 1, state - 2]
                if previous_score > best_previous_score:
                    best_previous_state = state - 2
                    best_previous_score = previous_score

            if torch.isfinite(best_previous_score):
                scores[frame, state] = (
                    best_previous_score
                    + log_probabilities[frame, current_token]
                )
                backpointers[frame, state] = best_previous_state

    # A complete CTC path may finish on the last token or the final blank.
    final_token_state = num_states - 2
    final_blank_state = num_states - 1
    if scores[-1, final_blank_state] > scores[-1, final_token_state]:
        final_state = final_blank_state
    else:
        final_state = final_token_state

    if not torch.isfinite(scores[-1, final_state]):
        raise ValueError('No valid CTC path reaches the complete target.')

    # Follow the saved choices backwards to recover one state per frame.
    state_path = [0] * num_frames
    current_state = final_state

    for frame in range(num_frames - 1, -1, -1):
        state_path[frame] = current_state
        if frame > 0:
            current_state = int(backpointers[frame, current_state])

    # Target tokens occupy odd positions in the expanded sequence:
    # blank, token 0, blank, token 1, blank, ...
    token_alignments = []
    for target_index, token_id in enumerate(target_ids):
        token_state = 2 * target_index + 1
        token_frames = [
            frame
            for frame, state in enumerate(state_path)
            if state == token_state
        ]

        # Reaching the final state guarantees that every token was visited.
        # Keep the check here so a future transition change fails loudly.
        if not token_frames:
            raise RuntimeError(
                f'Internal alignment error: target token {target_index} '
                'was not assigned any frame.'
            )

        token_alignments.append(
            TokenAlignment(
                target_index=target_index,
                token_id=token_id,
                start_frame=token_frames[0],
                end_frame=token_frames[-1] + 1,
            )
        )

    return CTCAlignment(
        log_score=float(scores[-1, final_state]),
        expanded_target=expanded_target,
        state_path=state_path,
        tokens=token_alignments,
    )
