'''Summarize exported handwriting boundaries by character pair.'''

import argparse
from pathlib import Path

from tva.boundary_statistics import (
    SUBSET_AGREEMENT_FILTERED,
    analyze_boundary_jsonl,
    write_boundary_statistics,
)


def analyze_boundaries(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / 'pair_analysis'
    )

    (
        summary,
        pair_rows,
        position_rows,
        region_pair_rows,
        region_position_rows,
    ) = analyze_boundary_jsonl(input_path)
    (
        summary_path,
        pair_path,
        overview_path,
        position_path,
        position_overview_path,
        region_pair_path,
        region_pair_overview_path,
        region_position_path,
        region_position_overview_path,
    ) = write_boundary_statistics(
        output_dir,
        summary,
        pair_rows,
        position_rows,
        region_pair_rows,
        region_position_rows,
        overwrite=args.overwrite,
    )

    quality_by_window = {
        window_ms: summary['global_statistics'][str(window_ms)][
            SUBSET_AGREEMENT_FILTERED
        ]['occurrence_count']
        for window_ms in summary['window_sizes_ms']
    }

    print('\nBoundary analysis complete')
    print(f'  boundaries: {summary["total_boundaries"]}')
    print(f'  unique pairs: {summary["unique_pairs"]}')
    print(f'  writers: {summary["total_writers"]}')
    print('  agreement/non-padding/unclipped counts:')
    for window_ms, count in quality_by_window.items():
        rate = count / summary['total_boundaries']
        print(f'    {window_ms:>3} ms: {count} ({rate:.1%})')

    region_statistics = summary['candidate_region_statistics']
    if region_statistics is not None:
        region_filtered = region_statistics[SUBSET_AGREEMENT_FILTERED]
        region_count = region_filtered['occurrence_count']
        region_rate = region_count / summary['total_boundaries']
        fallback_rate = region_filtered['fallback_window_rate']
        readable_fallback_rate = (
            f'{fallback_rate:.1%}'
            if fallback_rate is not None
            else 'n/a'
        )
        print('  complete candidate-region features:')
        print(
            f'    reliable: {region_count} ({region_rate:.1%}); '
            f'fallback used: {readable_fallback_rate}'
        )

    print('  100 ms filtered position diagnostics:')
    print('    timing reference is evenly spaced and is not ground truth')
    for position in ('only', 'first', 'middle', 'final'):
        row = next(
            item for item in position_rows
            if item['group_type'] == 'boundary_position'
            and item['group_value'] == position
            and item['window_ms'] == 100
            and item['subset'] == SUBSET_AGREEMENT_FILTERED
        )
        rate = row['no_low_force_rate']
        readable_rate = f'{rate:.1%}' if rate is not None else 'n/a'
        estimated = row['boundary_center_relative_median']
        reference = row['uniform_reference_relative_median']
        relative_offset = row['boundary_relative_offset_median']
        time_offset = row['boundary_time_offset_ms_median']
        readable_timing = 'n/a'
        if relative_offset is not None:
            readable_timing = (
                f'est={estimated:.1%}, ref={reference:.1%}, '
                f'offset={relative_offset:+.1%} '
                f'({time_offset:+.0f} ms)'
            )
        print(
            f'    {position:>6}: {row["occurrence_count"]} '
            f'({readable_rate} contact preserved; {readable_timing})'
        )
    print(f'  JSON report: {summary_path}')
    print(f'  pair overview: {overview_path}')
    print(f'  pair statistics: {pair_path}')
    print(f'  position overview: {position_overview_path}')
    print(f'  position statistics: {position_path}')
    if region_pair_overview_path is not None:
        print(f'  region pair overview: {region_pair_overview_path}')
        print(f'  region pair statistics: {region_pair_path}')
        print(f'  region position overview: {region_position_overview_path}')
        print(f'  region position statistics: {region_position_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Calculate threshold-free global and per-pair statistics from a '
            'boundary JSONL export.'
        )
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to boundaries.jsonl.',
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory. Defaults to pair_analysis beside the input.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing analysis in the output directory.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    analyze_boundaries(parse_args())
