# LLM Explanation - Project Instructions

## What this project is

Investigating whether LLMs can explain algorithmic resource-allocation decisions to
users with different backgrounds and priorities. See `Project - LLM Explanation.md` for
the full description. We build around case studies: a simulator plus a decision system,
then layered explanation mechanisms, then evaluation.

## Folder map

- `Project - LLM Explanation.md` - project hub.
- `Case Studies/<name>/` - one folder per case study. Each has its own `CLAUDE.md`,
  a `persona/` and `history/` folder, and `Run - *.md` / `Summary - *.md` notes.
- `Experiment Design/` - how each experiment is set up before it is run.

## Conventions

- Note naming follows the parent convention: `Run - ...`, `Summary - ...`, hub as
  `Project - ...`. Persona files use the standard fields (occupation, background,
  expertise, preferences, need level, ethics, disposition toward the mechanism).
- Keep links portable (audience may read on GitHub, not only Obsidian).
- Keep personal, unshared working docs under `Local/` (git-ignored). Do not link to
  `Local/` paths from pushed files; teammates will not have them.
- Code does not live in this repo yet. When it does, decide repo location first
  (iCloud sync and git internals do not mix well) and update `.gitignore`.

## Evaluation metrics

The evaluation metrics are still being developed and refined through our literature
review, so treat no list as final. The current candidate metrics, what each captures,
and how to measure them live in the working table at
`Experiment Design/Metrics - Explanation Evaluation.md`. Use that table when assessing an
explanation, and update it there rather than hard-coding metrics elsewhere.

## Hard constraint (from the toy study, applies project-wide)

Explanations must be honest. Do not try to convince an agent that an allocation is good
for them. Name defects plainly rather than spinning them.
