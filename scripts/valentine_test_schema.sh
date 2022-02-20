#!/bin/bash

basedir=/tank/local/suhail/data/relic/relic_datasets_vldb_2021/dataset_flat_exact/
#basedir=/tank/local/suhail/data/relic/real-world/
combined=''

counter=0

{
#for i in {1..16};
for f in "$basedir"*;
#for i in  10 11 12
do
  if [ -d ${f} ];
  then
    i=$(basename $f)
    echo Processing Workflow $i

    for matcher in sf ;
    do
      for alpha in 0.1 0.2 0.5 0.75 ;
      do
        for beta in 0.1 0.2 0.5 0.75;
        do
          for prefix in true false ;
          do
            outfile=$basedir/$i/$combined/inferred/valentine_${matcher}_perturb_${alpha}_${beta}_result.csv
            python valentine_test.py $basedir $i $outfile $matcher $alpha $prefix $beta &
            counter=$((counter+1))

            if [ $counter -eq 30 ];
            then
              counter=0
              wait
            fi
          done
        done
      done
    done



  fi
done
} 2>&1 | tee -a $basedir/valentine_run.log

wait

echo "Valentine run completed at $basedir, Time: `date`" | mailx -s "Valentine Experiment" suhail@uchicago.edu
