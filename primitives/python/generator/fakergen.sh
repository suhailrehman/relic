#!/bin/bash

for ver in 20 50 # 100
do
  for col in 10 20 # 50
  do
    for row in 100 1000 10000 50000 100000
    do
      for iter in 1 2 3 4 5 6 7 8 9 10
      do
        python fakergen.py --output_dir=dataset_flat_new --col=$col --row=$row --ver=$ver  --npp=True
      done
    done
  done
done



#python fakergen.py --output_dir=dataset_flat_new --col=20 --row=1000 --ver=20  --npp=True

#python fakergen.py --output_dir=mattest --col=20 --row=1000 --ver=100 --matfreq=5 --npp=True



# Generate at different materialization frequencies
#for mat in 1 2 4 5
#do
#  for iter in 1 2 3 4 5
#  do
#    python fakergen.py --output_dir=mattestnpp --col=20 --row=1000 --ver=50 --matfreq=$mat --npp=True
#  done
#done
