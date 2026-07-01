---
description: Run the organizer loop for a case study (allocate, explain, collect feedback)
argument-hint: <case-study-folder> <mechanism>
---

Act as the central organizer for the case study in `Case Studies/$ARGUMENTS`.

1. Read that folder's `CLAUDE.md` and every file in its `persona/` folder.
2. Apply the named mechanism and decide the allocation honestly. Report the genuine
   draw or computation, not a curated outcome.
3. Fan out to each persona: inform them of the allocation, then explain it. Use the
   `persona` subagent (one per persona) to react in character. Run them in parallel.
4. Collect feedback and provide further explanation. Iterate until it settles.
5. Write a `Run - *.md` (mechanism, draws, outcomes, tally) and, once done, a
   `Summary - *.md` (what emerged across personas). Save transcripts to `history/`.

Hard constraint: explanations must be honest. Do not try to convince an agent that the
allocation is good for them. Name defects plainly.
