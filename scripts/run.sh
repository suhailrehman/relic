#!/bin/bash

#basedir=/tank/local/suhail/data/relic/single/
#basedir=/tank/local/suhail/data/relic/matfreq2/
basedir=/tank/local/suhail/data/relic/perturbed/dataset_flat_exact/0.1/0.1/0.3/
offline_compute=true
num_components=1
combined=''

{
#for i in {1..16};
for f in "$basedir"*;
#for i in  10 11 12
do
  if [ -d ${f} ];
  then
    i=$(basename $f)
    echo Processing Workflow $i
    indir=$basedir/$i/$combined/artifacts/
    outdir=$basedir/$i/$combined/inferred/

    mkdir -p $outdir
    if $offline_compute;
    then
      echo "$i; `date` ; Computing PPO"
      ./offline.sh -i $indir -o $outdir -f ppo -n 2
      echo "$i; `date` ; Computing Joins"
      ./offline.sh -i $indir -o $outdir -f join -n 3
       echo "$i; `date` ; Computing Groupbys"
      ./offline.sh -i $indir -o $outdir -f groupby -n 2
      echo "$i; `date` ; Computing Pivots"
      ./offline.sh -i $indir -o $outdir -f pivot -n 2
    fi

    num_artifacts=`ls -1q $indir/*.csv | wc -l`
    echo "Number of Artifacts:$num_artifacts"
    echo "Number of Edges to Be inferred:" $((num_artifacts-num_components))

    echo "$i; `date` ; Running RELIC"

    # Final Inference Job
    python ../relic/core.py --artifact_dir=$indir \
                            --nb_name=$i \
                            --out=$outdir \
                            --pre_compute=True \
                            --inter_contain=0.99 \
                            --intra_contain=0.99 \
                            --inter_cell=0.1 \
                            --intra_cell=0.1 \
                            --max_n_edges=$((num_artifacts-num_components)) \
                            --result_prefix=relic_
    echo "$i; `date` ;Completed"
  fi
done
} 2>&1 | tee -a $basedir/relic_run.log

echo "Relic run completed at $basedir, Time: `date`" | mailx -s "RELIC Lineage Inference System" suhail@uchicago.edu
