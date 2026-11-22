"""Build OnHW bigram vocabularies with the TVA command-line interface."""

from tva.force_bigram import build, parse_args, select_emitting_pairs

__all__ = ['build', 'parse_args', 'select_emitting_pairs']


if __name__ == '__main__':
    build(parse_args())
