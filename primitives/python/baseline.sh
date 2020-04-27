#!/bin/bash


BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset/"
#BASE_DIR="/home/suhail/ok/"
#BASE_DIR="/home/suhail/Projects/sample_workflows/million_notebooks/selected/"


for f in "$BASE_DIR"*; do
    if [ -d ${f} ]; then
        # Will not run if no directories are available
        nb_name=$(basename $f)

        python baseline.py --basedir=$BASE_DIR --nbname=$nb_name

      if [ $? -ne 0 ]
      then
        echo "Error in "+ $nb_name
        errors+=(nb_name "$errors[@]")
      fi
    fi
done


echo "All error notebooks: $errors"
