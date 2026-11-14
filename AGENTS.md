# Shared thesis instructions

These instructions apply to Codex and Claude Code in this repository. Work as
researcher, implementer, or reviewer as the task needs; no separate agents or
role framework are required.

## Start each session

1. Read [README.md](README.md) for the implementation and commands.
2. Read the current-status snapshot in [docs/PROGRESS.md](docs/PROGRESS.md)
   and the latest relevant entries.
3. Read [docs/DECISIONS.md](docs/DECISIONS.md), especially D026 for the current
   dataset policy and D024/D025 for frozen tokenizer construction.
4. For the task, consult [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
   [docs/ABLATION_PLAN.md](docs/ABLATION_PLAN.md),
   [docs/METHOD.md](docs/METHOD.md), and [docs/SETUP.md](docs/SETUP.md) as needed.
   Inspect relevant code, configs, tests, and artifact indexes before editing.
5. Check Git status and existing changes. Preserve work from the user or
   another session; do not overwrite or revert unrelated changes.

Older proposal and scope notes describe earlier stages. Some are local and
Git-ignored. They are historical context, not required entry points.
Use explicit user instructions and recorded decision supersessions to resolve
known differences. Flag any unresolved scope or method conflict before work
that depends on it. Do not infer progress or approval from an old to-do list.

## Repository responsibilities

TVA is the main thesis context. DTLR is a separate evidence-producing repository:

- **DTLR:** handwriting images and transcriptions -> character localization,
  alignment, ink-connectivity measurements -> versioned evidence export.
- **TVA:** that evidence and the declared construction corpus -> tokenizer
  vocabularies -> recognition training and evaluation.
- The evidence handoff records source inputs, schema, code revision, and
  checksums. Preserve frozen exports; validate new approved versions before
  using them. See [the evidence index](artifacts/evidence/README.md).
- For DTLR work, follow its [local instructions](../DTLR/AGENTS.md) and
  [technical handoff](../DTLR/PROJECT_HANDOFF.md). Those sibling links assume
  adjacent checkouts. TVA training needs the frozen evidence, not a DTLR
  checkout. Overall thesis progress and scientific decisions remain here.

## Explanation style

Use [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf)
writing principles and Cavemen-style simplicity for all explanations,
especially implementation details.

- Use short, complete sentences, common words, and active voice.
  Give one instruction or main idea per sentence.
- Assume the reader is new to the topic. Explain technical terms when first
  used. Start with the purpose, then explain how the parts work.
- For implementation, explain the input, the steps, and the output. State why
  a change is needed and how it was checked. Use a small example when helpful.
- Keep explanations direct and respectful. Preserve exact code identifiers,
  commands, numbers, research meaning, and uncertainty when simplifying.

## Workflow: Understand -> Plan -> Get Approval -> Execute -> Verify -> Update Progress

1. **Understand:** establish the research purpose, current state, relevant
   approved decisions, and requested outcome from repository evidence. Ask
   about material gaps instead of guessing.
2. **Plan:** briefly state the intended change, affected files, scientific
   implications, and verification. Keep the plan proportional to the task.
3. **Get approval:** wait for explicit approval before changing the research
   question or scope, methodology, datasets, evaluation metrics, experimental
   design, or substantial architecture/implementation. Also obtain approval
   before deleting work or making destructive changes. A proposal is not an
   approved decision.
   Small implementation tasks within the requested or approved scope may
   proceed without repeated approval. Reuse approval already given for the
   same plan; ask again only if the work would exceed it.
4. **Execute:** make the smallest change that satisfies the approved task.
   Preserve working code, project structure, frozen artifacts, and historical
   results. Do not add dependencies, scripts, refactors, or orchestration
   merely to support this workflow.
5. **Verify:** run checks appropriate to the change using the existing
   environment and tools. Inspect actual outputs; do not equate writing code
   or starting a run with completion. Report commands, outcomes, skipped
   checks, and limitations. If verification fails, report it and fix within
   scope or explain the blocker.
6. **Update progress:** record meaningful outcomes, evidence, unresolved
   issues, and next steps in the existing documents below. Finish with a
   concise handoff stating what changed and what was verified.

## Research and experiment safeguards

- Tokenization changes output text labels; preserve each dataset's approved
  sensor inputs and recognition protocol. Follow D026 rather than treating
  historical OnHW settings as ED settings.
- Respect the frozen construction in D024/D025. A setting described as an
  available ablation is not automatic approval to add that experiment.
- Keep comparison conditions matched except for the predeclared factor.
  Preserve split assignments, seeds, preprocessing, training budgets,
  segmentation policies, and reporting rules unless an approved experiment
  explicitly changes them.
- Do not use held-out recognition labels, frequencies, or recognition scores
  to select vocabulary, connectivity thresholds, or evidence weights for the
  same evaluation. Declared task alphabets may supply character fallbacks.
- Preserve frozen tokenizer files and IDs, evidence provenance, saved configs,
  checkpoints, predictions, and logs. Do not overwrite completed runs or
  relabel historical results as current-method results.
- For new experiments, record the code revision and any uncommitted changes,
  resolved config, data/evidence and tokenizer versions, fold, seed, output
  path, and checkpoint-selection rule. Use existing checksums when available.
- Follow the current metric rule in the experiment overview. Keep single-fold
  development findings separate from final multi-fold results. Distinguish
  same-checkpoint CER/WER from independently selected minima.
- Never fabricate citations, quotations, results, or experimental findings.
  Verify sources before citing them; record a real source link or identifier.
  Separate observed evidence, interpretation, and hypotheses. Label unverified
  claims and missing evidence explicitly; retain negative and failed results.

## Verification guidance

Use the active environment documented in [docs/SETUP.md](docs/SETUP.md);
check the actual interpreter and required local data before running commands.

- Documentation: review the diff, relative links, and consistency with current
  decisions; run `git diff --check`.
- Implementation: run relevant existing tests and a bounded smoke check when
  needed. Tokenizer checks require `PYTHONHASHSEED=42`; the existing suite can
  be run with `PYTHONHASHSEED=42 python -m pytest -q`.
- Experiments: validate configs, frozen inputs, and output paths before
  execution; use the existing launcher's dry run where supported. Confirm
  completion and inspect saved metrics/predictions before recording results.
- Do not launch full training to verify a documentation edit or small fix.
  If data, dependencies, or compute are unavailable, state what remains
  unverified rather than claiming success.

## Commit and push

When the user says "commit and push", commit the approved changes and push the
current branch to both configured repositories: GitHub (remote `origin`) and
GitLab (remote `gitlab`). This request authorizes both pushes unless the user
explicitly limits the destination.

Check the actual remote destinations before pushing. Verify that both remote
branch tips match the local commit afterward. If either push fails, report
which destination failed and what remains; do not claim both are synchronized.
Do not force-push, rewrite published history, or include unrelated work
without explicit authorization.

## Maintain the existing record

- [docs/PROGRESS.md](docs/PROGRESS.md): keep the top snapshot concise and
  current (completed work, active task, open issues, next steps). Insert dated
  milestones directly below the snapshot, newest first. For the same date,
  put the newest update first. Preserve all older entries.
- [docs/DECISIONS.md](docs/DECISIONS.md): append important approved decisions
  with status, rationale, and what they supersede. Mark pending proposals as
  pending. Preserve earlier decisions as history.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): record actual scientific runs,
  provenance, results, and caveats; link detailed evidence from progress.

Update only the documents affected by meaningful progress, not every file
after every small edit. Link to existing evidence instead of duplicating it.
At a handoff, distinguish completed work from planned work and record the
next concrete action and any approval still needed.
