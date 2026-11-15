from dataclasses import dataclass
from typing import Callable

import torch

from .ctc_alignment import CTCAlignment

__all__ = [
    'AlignmentAnalysis',
    'BoundaryRegion',
    'CharacterDiagnostic',
    'analyze_alignment',
]


@dataclass(frozen=True)
class CharacterDiagnostic:
    '''Alignment position and model confidence for one target character.'''

    target_index: int
    token_id: int
    text: str
    model_start_frame: int
    model_end_frame: int
    input_start_sample: int
    input_end_sample: int
    anchor_input_sample: float
    aligned_probability: float
    preferred_token_id: int
    preferred_text: str
    preferred_probability: float
    best_competing_probability: float
    confidence_margin: float
    agrees_with_greedy: bool
    overlaps_padding: bool


@dataclass(frozen=True)
class BoundaryRegion:
    '''Candidate region between two adjacent character-emission anchors.'''

    boundary_index: int
    left_text: str
    right_text: str
    pair: str
    model_start_frame: int
    model_end_frame: int
    input_start_sample: int
    input_end_sample: int
    center_input_sample: float
    num_blank_frames: int
    duration_ms: float


@dataclass(frozen=True)
class AlignmentAnalysis:
    '''Character diagnostics and candidate boundary regions for one sample.'''

    num_model_frames: int
    num_model_input_samples: int
    num_raw_samples: int
    modeled_input_samples: int
    trailing_unmodeled_samples: int
    downsampling_ratio: int
    sample_rate_hz: float
    characters: list[CharacterDiagnostic]
    boundaries: list[BoundaryRegion]


def _input_position(
    model_frame: int,
    downsampling_ratio: int,
    num_model_input_samples: int,
) -> int:
    '''Map an output-frame edge to an approximate model-input position.'''
    return min(model_frame * downsampling_ratio, num_model_input_samples)


def analyze_alignment(
    alignment: CTCAlignment,
    probabilities: torch.Tensor,
    token_to_text: Callable[[int], str],
    downsampling_ratio: int,
    num_model_input_samples: int,
    num_raw_samples: int,
    sample_rate_hz: float,
) -> AlignmentAnalysis:
    '''Add interpretable confidence and approximate input positions.

    The character positions are emission anchors, not exact physical stroke
    boundaries. Probabilities are diagnostics from the trained model and are
    not assumed to be perfectly calibrated confidence values.
    '''
    if probabilities.ndim != 2:
        raise ValueError(
            'probabilities must have shape (num_frames, num_classes).'
        )
    if probabilities.shape[0] != len(alignment.state_path):
        raise ValueError('probabilities and alignment have different lengths.')
    if downsampling_ratio <= 0:
        raise ValueError('downsampling_ratio must be positive.')
    if num_model_input_samples <= 0 or num_raw_samples <= 0:
        raise ValueError('sample counts must be positive.')
    if sample_rate_hz <= 0:
        raise ValueError('sample_rate_hz must be positive.')

    probabilities = probabilities.detach().to(device='cpu', dtype=torch.float64)
    characters = []

    for token in alignment.tokens:
        span = probabilities[token.start_frame : token.end_frame]
        mean_distribution = span.mean(dim=0)

        aligned_probability = float(mean_distribution[token.token_id])
        preferred_token_id = int(mean_distribution.argmax())
        preferred_probability = float(mean_distribution[preferred_token_id])

        competitors = mean_distribution.clone()
        competitors[token.token_id] = -torch.inf
        best_competing_probability = float(competitors.max())

        input_start = _input_position(
            token.start_frame,
            downsampling_ratio,
            num_model_input_samples,
        )
        input_end = _input_position(
            token.end_frame,
            downsampling_ratio,
            num_model_input_samples,
        )

        characters.append(
            CharacterDiagnostic(
                target_index=token.target_index,
                token_id=token.token_id,
                text=token_to_text(token.token_id),
                model_start_frame=token.start_frame,
                model_end_frame=token.end_frame,
                input_start_sample=input_start,
                input_end_sample=input_end,
                anchor_input_sample=(input_start + input_end) / 2,
                aligned_probability=aligned_probability,
                preferred_token_id=preferred_token_id,
                preferred_text=token_to_text(preferred_token_id),
                preferred_probability=preferred_probability,
                best_competing_probability=best_competing_probability,
                confidence_margin=(
                    aligned_probability - best_competing_probability
                ),
                agrees_with_greedy=(preferred_token_id == token.token_id),
                overlaps_padding=(input_end > num_raw_samples),
            )
        )

    boundaries = []
    for boundary_index, (left, right) in enumerate(
        zip(characters, characters[1:])
    ):
        start_frame = left.model_end_frame
        end_frame = right.model_start_frame
        num_blank_frames = max(0, end_frame - start_frame)
        input_start = _input_position(
            start_frame,
            downsampling_ratio,
            num_model_input_samples,
        )
        input_end = _input_position(
            end_frame,
            downsampling_ratio,
            num_model_input_samples,
        )

        boundaries.append(
            BoundaryRegion(
                boundary_index=boundary_index,
                left_text=left.text,
                right_text=right.text,
                pair=left.text + right.text,
                model_start_frame=start_frame,
                model_end_frame=end_frame,
                input_start_sample=input_start,
                input_end_sample=input_end,
                center_input_sample=(input_start + input_end) / 2,
                num_blank_frames=num_blank_frames,
                duration_ms=(
                    num_blank_frames
                    * downsampling_ratio
                    / sample_rate_hz
                    * 1000
                ),
            )
        )

    modeled_input_samples = min(
        probabilities.shape[0] * downsampling_ratio,
        num_model_input_samples,
    )

    return AlignmentAnalysis(
        num_model_frames=probabilities.shape[0],
        num_model_input_samples=num_model_input_samples,
        num_raw_samples=num_raw_samples,
        modeled_input_samples=modeled_input_samples,
        trailing_unmodeled_samples=max(
            0, num_model_input_samples - modeled_input_samples
        ),
        downsampling_ratio=downsampling_ratio,
        sample_rate_hz=sample_rate_hz,
        characters=characters,
        boundaries=boundaries,
    )
