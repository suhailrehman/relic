#!/bin/bash


BASE_DIR="/home/suhail/ok/"

for f in "$BASE_DIR"*; do
    if [ -d ${f} ]; then
        # Will not run if no directories are available
        nb_name=$(basename $f)

	mkdir -p $f/artifacts
	mv $f/*.csv $f/artifacts/

        #python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering=PC2

      if [ $? -ne 0 ]
      then
        echo "Error in "+ nb_name
        errors+=(nb_name "$errors[@]")
      fi
    fi
done


echo "All error notebooks:"+$errors
