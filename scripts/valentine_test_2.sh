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

    for matcher in coma jlm sf ;
    do
      outfile=$basedir/$i/$combined/inferred/valentine_${matcher}_result.csv
      python valentine_test.py $basedir $i  $outfile $matcher &
      counter=$((counter+1))
    done

    if [ $counter -eq 30 ];
    then
      counter=0
      wait
    fi

  fi
done
} 2>&1 | tee -a $basedir/valentine_run.log

wait

echo "Valentine run completed at $basedir, Time: `date`" | mailx -s "Valentine Experiment" suhail@uchicago.edu
