#!/bin/bash

#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset/"
BASE_DIR="/home/suhail/ok/"
#BASE_DIR="/home/suhail/Projects/sample_workflows/million_notebooks/selected/"

# Run on a list of notebooks
nb_list=`cat nblist.txt`

#for f in "$BASE_DIR"*; do
for f in $nb_list; do
    if [ -d ${f} ]; then
        # Will not run if no directories are available
        nb_name=$(basename $f)

        rm -rf $BASE_DIR$nb_name'/inferred/'

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --group=True

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --recompute=True

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=valset --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=rowvalset  --recompute=True

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colvalset --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cellvalset --recompute=True


        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cellcol --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2colcell --swap=True --recompute=True


      if [ $? -ne 0 ]
      then
        echo "Error in "+ nb_name
        errors+=(nb_name "$errors[@]")
      fi
    fi
done


echo "All error notebooks: $errors"
