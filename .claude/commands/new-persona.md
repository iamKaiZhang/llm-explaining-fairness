---
description: Scaffold a new agent persona for a case study
argument-hint: <case-study-folder> <persona-name>
---

Create a persona file at `Case Studies/<case-study-folder>/persona/<persona-name>.md`
for: $ARGUMENTS

Match the format of the existing personas. Include these fields as a markdown list:

- **Occupation**
- **Background** (age, formation, how they think about fairness)
- **Expertise**
- **Preferences** (what they want from the resource, how strongly)
- **Need level**
- **Ethics** (which fairness notion they lean toward)
- **Disposition toward the mechanism** (how they react to the allocation rule)

Rules:
- Give the persona a distinct fairness intuition so the set spans different notions.
- Do not invent sensitive real-world attributes beyond what the study needs.
- If the target case study or a needed detail is unclear, ask before writing.
