# Ablation Study Plan

This document explains what an ablation study is, why we need one, and how we
plan to run it for this project. It is written so that anyone, including
people new to the project, can follow it.

## The project in one paragraph

We want to recognize handwriting better by splitting text into pieces
("tokens") in a smarter way. Most tokenizers decide which letters belong
together by looking only at text: letter pairs that appear often become one
token. Our idea is different: we look at real handwriting and check which
letter pairs people actually **write connected**, without lifting the pen.
Those pairs become tokens.

The work happens in two steps:

1. **DTLR** looks at handwriting images (IAM), finds each character, and
   checks whether neighbouring characters are joined by ink. The result is
   the **connectivity evidence**: for each letter pair, how often it is
   written connected.
2. **TVA** uses that evidence to build **handwriting-aware tokenizers**
   (Bigram, BPE, Unigram) and then trains the handwriting recognition model
   with them.

## What is an ablation study?

An ablation study answers one question: **which parts of our system are
really responsible for the result?**

The idea is simple. Take the full system, change or remove **one part at a
time**, keep everything else exactly the same, and see what happens to the
error rate.

- If the error gets worse when a part is removed, that part was helping.
- If nothing changes, that part was not doing much.

A simple analogy: if a cake tastes great, you can bake it again without the
sugar, then again without the vanilla, and so on. Whatever makes the cake
taste worse when you leave it out is what made it good.

## Why we need one

So far, our handwriting-aware tokenizers have a lower character error rate
than the text-only tokenizers in every fold-0 comparison (see
`docs/EXPERIMENTS.md`, 2026-10-23 and 2026-11-01). The gains are bigger in the
writer-dependent split than the writer-independent one, and in one case (BPE,
writer-independent) the word error rate is actually slightly worse. That is
encouraging, but it does not yet prove **why** they win. Some other
explanations are still possible:

- They might win just because they merge fewer letters (different
  "compression"), not because of connectivity.
- They might win because of the vocabulary size, or how they were trained.
- The difference might simply be luck, since every result so far comes from
  **a single fold and a single run**.

An ablation study lets us rule these out one by one, so we can honestly say
"connectivity is the reason".

One more honest note: on fold 0, the plain character model (no subword
tokens at all) is still better than every subword tokenizer. So our claim
should be: **connectivity helps subword tokenizers compared with text-only
ones**, not that it beats characters.

## Ground rules

These rules keep the study fair and easy to trust.

1. **One reference setup.** We pick one "anchor" setup and compare every
   experiment against it. Our anchor is the handwriting Bigram tokenizer:
   greedy splitting, 145 learned pieces, a pair must be seen at least 20
   times (`min_count = 20`), connectivity method `dominant-core-v3`, and the
   IAM-finetuned DTLR model.
2. **Change one thing at a time.** Every experiment differs from the anchor
   in exactly one setting. Everything else, including the recognition model
   and its training settings, stays the same.
3. **Repeat every experiment.** Run each one on all 5 data folds (and, if
   possible, more than one random seed). Report the average and how much it
   varies. A difference smaller than the normal variation is not a real
   difference.
4. **Measure the same way every time.** Report the character error rate
   (CER) and word error rate (WER) from the **same** training epoch: the one
   with the best validation CER. (Today they are sometimes taken from
   different epochs.)
5. **Say what we expect first.** Before running an experiment, write down
   what result we expect and what it would mean. This stops us from
   explaining results after the fact.

## What we have already done

Some of our existing experiments are already ablations. We just didn't call
them that.

| Experiment folder | The question it answers |
| --- | --- |
| `subword_comparison` | Does connectivity help at all? (with vs without evidence) |
| `letters_only` | Is the result only because text-only bigrams include spaces and punctuation? |
| `compression_matched` | Is the result only because handwriting tokenizers merge fewer letters? |
| `size_and_compression_matched` | Once size and compression are equal, does the *choice* of letter pairs still matter? |
| ~~`unigram_trainer`~~ | Retired on 2026-11-01: both sides now use TVA's own trainer (D024), so the training method can no longer differ |

These all live in `configs/thesis/ed/`. Most of them only have fold 0 results
so far.

## The plan, step by step

### Phase 0: Make the baseline solid (do this first)

- Fix the rules above (anchor setup, one measuring rule).
- Run all 5 folds for the main comparison (`subword_comparison`). The
  character baseline already has 5 folds.
- **Why:** this tells us how much results normally jump around between
  folds. Without it, we cannot tell whether later differences are real.

### Phase 1: Finish the checks we already built

- Run `compression_matched`, the three `drawn` bigram sets and
  `letters_only` on 5 folds (or 3 if computing time is short).
- **Why:** these are our main proof that connectivity itself, not
  compression or size, explains the result.

### Phase 2: Test the evidence itself (the most convincing part)

Here we deliberately make the connectivity evidence worse or flip it, and
see whether the recognizer suffers.

| Experiment | What we do | What we expect if connectivity really matters |
| --- | --- | --- |
| Reversed evidence | Pick the *least* connected letter pairs instead of the most connected | Clearly worse results |
| Noisy evidence | Randomly flip 10%, 25% and 50% of the "connected / not connected" labels | Results get steadily worse as noise grows |
| Less evidence | Build the evidence from 10%, 25%, 50% and 100% of the IAM lines | Shows how much handwriting data we actually need |
| Weaker connectivity method | Use the older `box-intersection-v1` method, which we know makes mistakes | Somewhat worse results |
| Weaker character finder | Use the DTLR model that was not fine-tuned on IAM | Somewhat worse results |

**Saving time:** before training anything, check how much each new
tokenizer differs from the anchor. If it shares almost all of its tokens with
the anchor, we can skip training it, because the result will be nearly the
same.

### Phase 3: Check sensitivity to settings

- **Vocabulary size:** try, for example, 50, 100, 145 and 200 learned pieces
  for both handwriting and text-only tokenizers. A curve says much more than
  a single point. Limits: the handwriting bigram can have at most 389 pieces
  (only 389 letter pairs are seen 20 times), and Unigram stops at about 470
  classes.
- **Number of allowed pairs** (BPE and Unigram): try 100, 145 and 200 most
  connected pairs (connected at least about 55%, 50% and 44% of the time).
- **Minimum count:** try `min_count` of 5, 10, 20 and 40.
- **Splitting policy** (bigram): greedy vs `dp`, always on both sides
  (config key `bigram_dp`).
- **Why:** this shows our result is not just a lucky choice of settings.

### Phase 4 (optional): Other handwriting sources

- Build the evidence from German handwriting (READ), or IAM and READ
  together, and see if it still helps on our English data.
- **Why:** this tests whether the idea carries over, not only whether it
  works once.

## Getting the code ready

Tokenizer construction is frozen (`docs/DECISIONS.md`, D024 and D025): every
tokenizer is built by TVA's own trainer, and a handwriting tokenizer only adds
a list of letter pairs allowed to join. Every experiment in Phases 2 and 3
changes only a setting: the allowed list, the vocabulary size or the
splitting policy. No construction code needs to change.

To run many experiments without mistakes, the pipeline needs a few
additions:

- **Named evidence versions.** Save each version of the evidence in its own
  folder (for example `evidence/<version>/connectivity.json.gz`) with a small
  record of how it was made: method, DTLR model, settings, data share, noise
  seed.
- **Generated configs.** Create the experiment configs from one base config
  plus small changes, instead of writing each YAML file by hand. Phases 0–3
  need well over 100 configs. The test `tests/test_ed_configs.py` should check
  that each config differs from the anchor in exactly one setting.
- **One launcher for folds and seeds.** Extend `train_cv.py` so any
  experiment can run on several folds and seeds with one command.
- **One results script.** Collect all finished runs into one table per
  experiment, showing the average, the variation, and the difference from the
  anchor.

## How much computing time?

Each experiment costs roughly:

> number of settings × 2 (writer-dependent and writer-independent) × number
> of folds × number of seeds

Phase 0 alone is about 60 training runs. If that is too much, a sensible
shortcut is to run Phases 2 and 3 on the writer-independent split only, with
3 folds.

## Summary

1. Make the main comparison reliable with 5 folds.
2. Finish the checks that rule out other explanations.
3. Damage the evidence on purpose and show that results get worse.
4. Show that the result holds across reasonable settings.
5. Optionally, show that it carries over to other handwriting data.

If all of this holds, we can say with confidence that **knowing how people
connect letters when they write leads to better tokenizers**.
