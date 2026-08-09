# Case Studies

One folder per case study. Each pairs a simulator and decision system with the
explanation mechanisms under study.

Suggested layout for a case study folder:

```
<name>/
  CLAUDE.md          # what the study is and the organizer's rules
  persona/           # one file per agent persona
  history/           # explanation transcripts, one file per agent
  Run - *.md         # a single run (mechanism, draws, outcomes)
  Summary - *.md     # what emerged across runs
```

Current studies:

- `Toy - Apple Allocation/` - one apple repeatedly allocated to five personas under
  different mechanisms (coin flip, biased coin flip, karma).
