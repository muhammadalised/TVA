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

    summary, pair_rows = analyze_boundary_jsonl(input_path)
    summary_path, pair_path, overview_path = write_boundary_statistics(
        output_dir,
        summary,
        pair_rows,
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
    print(f'  JSON report: {summary_path}')
    print(f'  pair overview: {overview_path}')
    print(f'  pair CSV:    {pair_path}')


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
