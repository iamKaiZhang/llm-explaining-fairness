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
- Code does not live in this repo yet. When it does, decide repo location first
  (iCloud sync and git internals do not mix well) and update `.gitignore`.

## Evaluation dimensions

When assessing an explanation, score against: self-consistency, counterfactual
robustness, conciseness, sufficiency and necessity for motivating the decision, and
relevance to the specific user.

## Hard constraint (from the toy study, applies project-wide)

Explanations must be honest. Do not try to convince an agent that an allocation is good
for them. Name defects plainly rather than spinning them.
