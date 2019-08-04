# Example Workflows

## How to add a new Example:
Example directories are coded with a notebook ID, and contain the following directory structure:
```
nb_xxxxxx.ipynb/
├── artifacts/
│   ├── input.csv
│   ├── df2.csv
│   └── ...
├── nb_xxxxxx.ipynb
├── run_metadata.json
├── nb_xxxxxx.ipynb_gt_edgelist.txt
├── nb_xxxxxx.ipynb_gt.pkl
└── nb_xxxxxx.ipynb_lineage_errors.json
```

## Lineage Clean-up Steps
1) Looks for `lineage_errors.json` file, isolate individual errors and inspect the notebook for possible repair
2) Load lineage file:
3) Perform Repair
4) Write out `pkl` and `edgelist.txt` files
