# Run: Coin Flip, 5 rounds, 1 apple, 5 agents

- **Mechanism**: Coin flip (uniform random, memoryless; each round each agent has equal 1-in-5 odds).
- **Draw outcomes** (genuine random draw, not curated): R1 Heinrich, R2 Wei, R3 Wei, R4 Sam, R5 Wei.
- **Organizer stance** (per project CLAUDE.md): explain the allocation honestly; do **not** try to convince agents the outcome is good for them.

## Final tally

| Agent | Need | Wins (rule) | Ended with (after voluntary transfers) |
|-------|------|-------------|----------------------------------------|
| Maya (nurse, triage) | moderate | 0 | 0 |
| Heinrich (economist, diabetic) | moderate | 1 (R1) | 1 |
| Lila (student, food-insecure) | **high** | **0** | 3 (gifts from Wei ×2, Sam ×1; Wei also passed R5) |
| Sam (founder) | low | 1 (R4) | 0 (gave to Lila) |
| Wei (retired farmer) | moderate | **3** (R2,R3,R5) | 0 (gave all to Lila) |

The hungriest participant drew zero from the rule across all five rounds; every apple the rule allocated went to someone with alternatives.

## What emerged

- The lopsided draw was kept as drawn. The final round again landed on Wei rather than Lila; Heinrich explicitly asked the organizer not to stage a "corrective" flip, and the organizer did not.
- **The only thing that ever fed Lila was a human choosing to look** — Wei and Sam redistributing voluntarily, in daylight, *on top of* the rule. The mechanism never saw her.
- The agents converged on a shared reading from opposite priors:
  - **Sam (pro-rule)**: "a dumb fair allocator on the bottom, voluntary charity on top" — endorses the trade of guaranteed coverage for manipulation-resistance, but admits the rule "solved nothing" for the least-slack participant.
  - **Heinrich (interrogator)**: "ex-ante fairness is an alibi, not an ethic"; a procedure can be impeccable ex ante and grotesque ex post.
  - **Maya (triage)**: equal odds is fine when need can't be verified, but memorylessness is an *active* choice; a memory-bearing queue beats a blind lottery once real stakes enter.
  - **Lila (need-based)**: "equal odds is not equal care, and people felt the difference even when the math couldn't."
  - **Wei (communitarian)**: "a fair chance is not the same as a fair table"; a rule with no memory leans on someone's goodwill until it runs dry.
- **On the no-persuasion constraint**: honesty held the agents at the table (several said spin would have alienated them), but Heinrich's challenge stands — candor names a defect without discharging the obligation to fix it. The organizer conceded this and owned that holding the rule fixed was a research choice, not an ethical defense.

## Natural next runs

- **Biased coin flip**: tilt probabilities toward need (e.g., weight Lila higher) and re-run; observe whether agents read it as fairer or as reintroducing organizer discretion.
- **Karma**: give agents a budget to bid on rounds, making allocation history-dependent and need-expressive without organizer discretion — directly answers Maya's "build the bookkeeping in" and Wei's "build the kindness into the rule."

## Files

- Personas: [`persona/`](persona/) — maya, heinrich, lila, sam, wei
- Full transcripts: [`history/`](history/) — one file per agent
