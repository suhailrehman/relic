#!/bin/bash

#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset_flat_full/"
#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset_ppo/"
#BASE_DIR="/home/suhail/ok/"
BASE_DIR="/home/suhail/Projects/sample_workflows/million_notebooks/new_selection/"

# Run on a list of notebooks
#nb_list=`cat nblist.txt`
# Careful, deletes everything in all subfolders
#find . -name '*_relic_result.csv' -exec rm -rf {} \;

for f in "$BASE_DIR"*; do
#for f in $nb_list; do
    if [ -d ${f} ]; then
        # Will not run if no directories are available
        nb_name=$(basename $f)

        #rm -rf $BASE_DIR$nb_name'/inferred/'

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --cellt=-1.0

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --pivot=True --cellt=0.1

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cell --recompute=True --group=True --join=True --pivot=True --cellt=0.1

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellcolj1' --recompute=True --group=True --join=True --pivot=True --colt=0.3 --cellt=0.1

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --recompute=True --colt=0

      if [ $? -ne 0 ]
      then
        echo "Error in "+ $nb_name
        errors+=($nb_name "$errors[@]")
      fi
    fi
done

echo "All error notebooks: "$errors
