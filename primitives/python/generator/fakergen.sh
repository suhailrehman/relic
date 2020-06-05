#!/bin/bash

#python fakergen.py --output_dir=dataset --col=20 --row=100 --ver=20
#python fakergen.py --output_dir=dataset --col=50 --row=100 --ver=20
#python fakergen.py --output_dir=dataset --col=10 --row=1000 --ver=20
#python fakergen.py --output_dir=dataset --col=10 --row=100 --ver=20
#python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=20
#python fakergen.py --output_dir=dataset --col=50 --row=1000 --ver=20
#python fakergen.py --output_dir=dataset --col=10 --row=100 --ver=50
#python fakergen.py --output_dir=dataset --col=20 --row=100 --ver=50
#python fakergen.py --output_dir=dataset --col=50 --row=100 --ver=50
#python fakergen.py --output_dir=dataset --col=10 --row=1000 --ver=50
#python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=50
#python fakergen.py --output_dir=dataset --col=50 --row=1000 --ver=50
#python fakergen.py --output_dir=dataset --col=10 --row=100 --ver=100
#python fakergen.py --output_dir=dataset --col=20 --row=100 --ver=100
#python fakergen.py --output_dir=dataset --col=50 --row=100 --ver=100
#python fakergen.py --output_dir=dataset --col=10 --row=1000 --ver=100
#python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=100
#python fakergen.py --output_dir=dataset --col=50 --row=1000 --ver=100


#python fakergen.py --output_dir=dataset --col=50 --row=100 --ver=20
#python fakergen.py --output_dir=dataset --col=50 --row=1000 --ver=100
#python fakergen.py --output_dir=dataset --col=20 --row=100 --ver=20
#python fakergen.py --output_dir=dataset --col=20 --row=100 --ver=20
#python fakergen.py --output_dir=dataset --col=20 --row=100 --ver=50
#python fakergen.py --output_dir=dataset --col=10 --row=100 --ver=50
#python fakergen.py --output_dir=dataset --col=10 --row=100 --ver=50
#python fakergen.py --output_dir=dataset --col=10 --row=100 --ver=100
#python fakergen.py --output_dir=dataset --col=20 --row=100 --ver=100
#python fakergen.py --output_dir=dataset --col=50 --row=100 --ver=100

#20200210-165524
#20200210-165531


#python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=100 --matfreq=2
#python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=100 --matfreq=4
#python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=100 --matfreq=5
#python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=100 --matfreq=10


#for ver in 20 50 # 100
#do
#  for col in 10 20 # 50
#  do
#    for row in 100 1000 #10000
#    do
#      for iter in 1
#      do
#        python fakergen.py --output_dir=dataset --col=$col --row=$row --ver=$ver  --npp=True
#      done
#    done
#  done
#done



python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=20  --npp=True

#for mat in 1 2 4 5 10
#do
#  for iter in 1 2 3 4 5
#  do
#    python fakergen.py --output_dir=dataset --col=20 --row=1000 --ver=100 --matfreq=$mat
#  done
#done
