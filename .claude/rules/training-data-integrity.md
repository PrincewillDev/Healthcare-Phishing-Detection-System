---
paths:
  - "src/preprocessing/**"
  - "src/features/**"
  - "src/models/**"
---
Any script that loads data from merged_raw.csv or any file derived
from it must filter to included_in_training == True before doing
anything else with the data (sampling, splitting, feature extraction,
training). Rows with included_in_training == False must never appear
in any training, validation, or test set, at any stage, under any
circumstance. If a script needs the excluded rows specifically (e.g.
for a post-training robustness check), that must be a separate,
explicitly named script, never mixed into the main pipeline.