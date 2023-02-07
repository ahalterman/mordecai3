## Creating training data for Mordecai 3

This directory contains raw training data and scripts to transform it into the
format that Mordecai 3 expects for its training.


- `synth_raw/`: synthetic training data for capitals and regions within countries
- `Pragmatic-Guide-to-Geoparsing-Evaluation/`: data from Gritta, Pilehvar, and Collier (2019) in LREV
- `orig_mordecai`: hand-labeled data collected as part of work on Mordecai 2.

These are transformed into 

- `all_loc_types/`
- `pa_only/`: Only populated places and administrative areas (no geographic features)