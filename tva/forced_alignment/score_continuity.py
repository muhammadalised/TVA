'''Create a provisional, corrected force-continuity pair ranking.'''

import argparse
from pathlib import Path

from tva.continuity_scoring import (
    DEFAULT_LOCAL_WINDOW_MS,
    DEFAULT_MIN_OCCURRENCES,
    DEFAULT_MIN_STRATUM_OCCURRENCES,
    DEFAULT_MIN_WRITERS,
    LOCAL_WEIGHT,
    REGION_WEIGHT,
    score_boundary_jsonl,
    write_continuity_scores,
)


def score_continuity(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / 'continuity_scores'
    )
    summary, rows = score_boundary_jsonl(
        input_path,
        local_window_ms=args.local_window_ms,
        min_occurrences=args.min_occurrences,
        min_writers=args.min_writers,
        min_stratum_occurrences=args.min_stratum_occurrences,
        local_weight=args.local_weight,
        region_weight=args.region_weight,
    )
    summary_path, score_path = write_continuity_scores(
        output_dir,
        summary,
        rows,
        overwrite=args.overwrite,
    )

    print('\nProvisional force-continuity scoring complete')
    print(
        f'  reliable boundaries: {summary["reliable_boundaries"]} / '
        f'{summary["total_boundaries"]}'
    )
    print(
        f'  eligible pairs: {summary["eligible_pairs"]} / '
        f'{summary["unique_scored_pairs"]}'
    )
    parameters = summary['parameters']
    print(
        f'  score: {parameters["local_weight"]:.0%} corrected '
        f'{parameters["local_window_ms"]} ms local contact + '
        f'{parameters["region_weight"]:.0%} corrected region contact'
    )
    print('  top eligible pairs:')
    for pair in summary['top_eligible_pairs'][:10]:
        print(
            f'    {pair["rank"]:>2}. {pair["pair"]:<4} '
            f'{pair["score"]:.3f} '
            f'(n={pair["occurrence_count"]}, '
            f'writers={pair["writer_count"]})'
        )
    print(f'  method report: {summary_path}')
    print(f'  pair ranking: {score_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Rank adjacent character pairs using a provisional force-only '
            'continuity score corrected for position and region duration.'
        )
    )
    parser.add_argument('--input', required=True, help='Path to boundaries.jsonl.')
    parser.add_argument(
        '--output-dir',
        help='Output directory. Defaults to continuity_scores beside the input.',
    )
    parser.add_argument(
        '--local-window-ms',
        type=int,
        default=DEFAULT_LOCAL_WINDOW_MS,
    )
    parser.add_argument(
        '--min-occurrences',
        type=int,
        default=DEFAULT_MIN_OCCURRENCES,
    )
    parser.add_argument(
        '--min-writers',
        type=int,
        default=DEFAULT_MIN_WRITERS,
    )
    parser.add_argument(
        '--min-stratum-occurrences',
        type=int,
        default=DEFAULT_MIN_STRATUM_OCCURRENCES,
    )
    parser.add_argument('--local-weight', type=float, default=LOCAL_WEIGHT)
    parser.add_argument('--region-weight', type=float, default=REGION_WEIGHT)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing score report in the output directory.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    score_continuity(parse_args())
