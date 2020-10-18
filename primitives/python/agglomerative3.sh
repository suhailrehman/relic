#!/bin/bash

#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset_flat/"
#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset_ppo/"
#BASE_DIR="/home/suhail/ok/"
BASE_DIR="/home/suhail/Projects/sample_workflows/million_notebooks/selected/"

# Run on a list of notebooks
#nb_list=`cat nblist.txt`

find . -name '*_relic_result.csv' -exec rm -rf {} \;

for f in "$BASE_DIR"*; do
#for f in $nb_list; do
    if [ -d ${f} ]; then
        # Will not run if no directories are available
        nb_name=$(basename $f)

        rm -rf $BASE_DIR$nb_name'/inferred/'


        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --cellt=-1.0
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --transform=True
        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --pivot=True --cellt=0.1
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cellcol --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cellcol --recompute=True --cellt=0 --colt=0

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cell --recompute=True --group=True --join=True --pivot=True --cellt=0.1
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cell --recompute=True --join=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cell --recompute=True --join=True --group=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cell --recompute=True --join=True --group=True --transform=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cellcol --recompute=True --join=True --group=True --transform=True --pivot=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cell --recompute=True --group=True --join=True  --pivot=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellcolgt' --recompute=True --group=True --colt=0.8
        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellcolj1' --recompute=True --group=True --join=True --pivot=True --colt=0.3 --cellt=0.1

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellcol' --recompute=True --join=True --group=True --pivot=True --colt=0.8 --transform=True

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellgt' --recompute=True --join=True --group=True --pivot=True --colt=0.8

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cellcol --recompute=True --join=True --group=True --transform=True --pivot=True

        python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --recompute=True --colt=0
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colgt --swap=True --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --recompute=True --group=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --recompute=True --join=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --recompute=True --join=True --group=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --recompute=True --join=True --group=True --transform=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --recompute=True --join=True --group=True --transform=True --pivot=True



        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=valset --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=rowvalset  --recompute=True

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colvalset --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cellvalset --recompute=True


        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2cellcol --recompute=True
        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2colcell --swap=True --recompute=True

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellgt' --recompute=True --join=True --group=True --pivot=True --colt=0.8

      if [ $? -ne 0 ]
      then
        echo "Error in "+ $nb_name
        errors+=($nb_name "$errors[@]")
      fi
    fi
done


echo "All error notebooks: "$errors
