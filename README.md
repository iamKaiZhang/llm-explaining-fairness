# LLM Explanation

Can LLMs explain algorithmic resource-allocation decisions to people with different
backgrounds and priorities: counterfactuals, prioritized metrics, and accessible yet
correct explanations? See [Project - LLM Explanation.md](Project%20-%20LLM%20Explanation.md)
for the full description.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Project - LLM Explanation.md` | Project hub and description |
| `Case Studies/` | One folder per case study (simulator + decision system + explanations) |
| `Experiment Design/` | How each experiment is set up before it is run |
| `.claude/` | Shared Claude Code commands, agents, and settings |

## Contributing

- Follow the note naming used across the project: `Run - ...`, `Summary - ...`, hub as
  `Project - ...`.
- Keep markdown links portable so notes render on GitHub as well as in Obsidian.
- Do not commit PDFs or any human-subjects data. Keep them out of the repo.
- Put secrets in a local `.env` (git-ignored). Never commit API keys.
- Code is not in this repo yet. The location is still to be decided.

## Working with Claude Code

Shared slash commands and agents live in `.claude/`. After cloning, they are available
automatically:

- `/new-persona` - scaffold a persona for a case study
- `/run-case-study` - run the organizer loop (allocate, explain, collect feedback)
- `/evaluate-explanation` - score an explanation against the evaluation dimensions
