---
name: Data Handling
description: Rules for handling data files and preprocessing scripts
paths:
  - "data/**"
  - "src/preprocessing/**"
---
Never commit raw dataset files (data/raw/**) to git. Add to .gitignore.

Any healthcare-context email sample that is adapted or synthesized
from a public dataset must be tagged as synthetic in its metadata.
Never present adapted samples as real healthcare breach data anywhere
in code, comments, or output.

Do not call external APIs (WHOIS, threat intel, live domain lookups)
inside the inference path. Use cached or offline data only, to keep
inference latency predictable.

