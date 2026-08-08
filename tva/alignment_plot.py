from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from .alignment_analysis import AlignmentAnalysis

__all__ = ['plot_alignment']


# Channel groups follow the dataset metadata order: AF, AR, G, M, F.
SENSOR_CHANNELS = {
    'AF': slice(0, 3),
    'AR': slice(3, 6),
    'G': slice(6, 9),
    'M': slice(9, 12),
    'F': 12,
}


def _shade_alignment(
    axes: list[Axes],
    analysis: AlignmentAnalysis,
) -> None:
    '''Draw candidate boundary regions and character emission anchors.'''
    rate = analysis.sample_rate_hz

    for boundary in analysis.boundaries:
        start_time = boundary.input_start_sample / rate
        end_time = boundary.input_end_sample / rate

        for axis in axes:
            if boundary.num_blank_frames:
                axis.axvspan(
                    start_time,
                    end_time,
                    color='tab:orange',
                    alpha=0.12,
                )
            else:
                axis.axvline(
                    start_time,
                    color='tab:orange',
                    alpha=0.45,
                    linestyle=':',
                    linewidth=1,
                )

    top_axis = axes[0]
    for character in analysis.characters:
        anchor_time = character.anchor_input_sample / rate
        color = 'tab:green' if character.agrees_with_greedy else 'tab:red'

        for axis in axes:
            axis.axvline(anchor_time, color=color, alpha=0.5, linewidth=1)

        top_axis.text(
            anchor_time,
            1.02,
            f'{character.text}\n{character.aligned_probability:.2f}',
            color=color,
            ha='center',
            va='bottom',
            fontsize=8,
            transform=top_axis.get_xaxis_transform(),
        )


def plot_alignment(
    raw_signal: np.ndarray,
    normalized_signal: np.ndarray,
    probabilities: torch.Tensor,
    analysis: AlignmentAnalysis,
    known_label: str,
    greedy_prediction: str,
    path_save: str | Path,
) -> None:
    '''Plot motion, raw force, model confidence, and aligned characters.'''
    if raw_signal.ndim != 2 or raw_signal.shape[1] != 13:
        raise ValueError('raw_signal must have shape (num_samples, 13).')
    if normalized_signal.ndim != 2 or normalized_signal.shape[1] != 13:
        raise ValueError(
            'normalized_signal must have shape (num_samples, 13).'
        )

    path_save = Path(path_save)
    path_save.parent.mkdir(parents=True, exist_ok=True)

    rate = analysis.sample_rate_hz
    num_raw_samples = raw_signal.shape[0]
    time_raw = np.arange(num_raw_samples) / rate

    # Ignore any model-only zero padding when plotting physical sensor data.
    normalized_signal = normalized_signal[:num_raw_samples]
    time_normalized = np.arange(len(normalized_signal)) / rate

    motion_groups = {
        name: np.linalg.norm(normalized_signal[:, channels], axis=1)
        for name, channels in SENSOR_CHANNELS.items()
        if name in ('AF', 'AR', 'G')
    }

    figure, axes_array = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        dpi=160,
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1.2, 1.2]},
    )
    axes = list(axes_array)

    for name, magnitude in motion_groups.items():
        axes[0].plot(
            time_normalized,
            magnitude,
            label=f'{name} magnitude',
            linewidth=1,
        )
    axes[0].set_ylabel('Normalized\nmagnitude')
    axes[0].legend(loc='upper right', ncol=3, fontsize=8)
    figure.suptitle(
        f'CTC alignment: label="{known_label}", greedy="{greedy_prediction}"'
    )

    axes[1].plot(
        time_raw,
        raw_signal[:, SENSOR_CHANNELS['F']],
        color='tab:purple',
        linewidth=1,
    )
    axes[1].set_ylabel('F channel\n(raw units)')

    probabilities = probabilities.detach().cpu()
    frame_centers = (
        np.arange(probabilities.shape[0]) + 0.5
    ) * analysis.downsampling_ratio / rate
    axes[2].plot(
        frame_centers,
        probabilities.max(dim=1).values.numpy(),
        color='tab:blue',
        marker='.',
        linewidth=1,
        label='Highest class probability',
    )
    for character in analysis.characters:
        axes[2].scatter(
            character.anchor_input_sample / rate,
            character.aligned_probability,
            color=(
                'tab:green'
                if character.agrees_with_greedy
                else 'tab:red'
            ),
            s=22,
            zorder=3,
        )
    axes[2].set_ylabel('Model\nprobability')
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='lower right', fontsize=8)

    _shade_alignment(axes, analysis)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0, max(num_raw_samples / rate, 1 / rate))

    figure.text(
        0.01,
        0.01,
        (
            'Green anchor: forced character agrees with local model choice.  '
            'Red: target constraint overrides it.  Orange: blank-frame '
            'candidate boundary region.'
        ),
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(path_save)
    plt.close(figure)
