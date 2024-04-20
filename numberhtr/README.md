# Beam-Analysis

This folder contains a minimal version of the [SimpleHTR repository](https://github.com/githubharald/SimpleHTR), and more information about the files within can be found there.

## Usage

To train a new SimpleHTR model, change the `SOURCE` variable in all relevant files (`main.py`, `model.py`, and `preprocessor.py`) to the value `"home"`, and run (from the `src/` directory):

```
python3 main.py --mode train --fast --data_dir ../../data/number/ --batch_size 500 --early_stopping 10 --fast
```

Parameters can be adjusted freely. If running with `--fast`, `create_lmdb.py` must be run in the data directory first.
