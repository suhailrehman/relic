#!/bin/bash

#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset_flat_timing/"
#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/timing_test/"
#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset_ppo/"
#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/mattestnpp/"
#BASE_DIR="/home/suhail/ok/"
#BASE_DIR="/home/suhail/Projects/sample_workflows/million_notebooks/selected_1/"
#BASE_DIR='/home/suhail/Projects/sample_workflows/million_notebooks/new_selection/'
#BASE_DIR="/home/suhail/Projects/sample_workflows/million_notebooks/selected_timing/"
BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/mattestnpp2/"



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
        #rm "$BASE_DIR$nb_name"/*_relic_result.csv

        # Flat Cell Level
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --cellt=-1.0
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --cellt=0.1 --pivot=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric='cell+containment' --recompute=True --join=True --group=True --cellt=0.1 --pivot=True

        # Cluster + Cell
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cell' --recompute=True --group=True --join=True --cellt=0.1 --pivot=True
        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cell+containment' --recompute=True  --cellt=0.1 --pivot=True --group=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellcolj1' --recompute=True --group=True --join=True --colt=0.3 --cellt=0.1 --pivot=True

        # Column BaseLine and w/ Detectors
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --recompute=True --colt=0
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --recompute=True --group=True --join=True --colt=0.1 --cellt=0.1 --pivot=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2col' --swap=True --recompute=True --group=True --join=True --colt=0.1 --cellt=0.1 --pivot=True


        # Column Multiset
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colms --swap=True --recompute=True --colt=0
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colms --recompute=True --join=True --group=True --cellt=0.1 --pivot=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2colms --recompute=True --group=True --join=True --cellt=0.1 --pivot=True

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colmscon --swap=True --recompute=True --colt=0
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colmscon --recompute=True --join=True --group=True --cellt=0.1 --pivot=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2colmscon --recompute=True --group=True --join=True --cellt=0.1 --pivot=True

        # Cell Containment
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric='cc_con' --recompute=True --cellt=-1
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric='cc_con' --recompute=True --group=True --join=True  --cellt=0.1 --pivot=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cc_con' --recompute=True --group=True --join=True  --cellt=0.1 --pivot=True



        # Timing Runs

        #python agglomerative_full.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cell+containment' --group=True --join=True --cellt=0.1 --pivot=True
        #python agglomerative_full.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --colt=0



      if [ $? -ne 0 ]
      then
        echo "Error in "+ $nb_name
        errors+=($nb_name "$errors[@]")
      fi
    fi
done

echo "All error notebooks: "$errors
