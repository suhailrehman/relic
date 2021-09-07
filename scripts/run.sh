#!/bin/bash

basedir=/tank/local/suhail/data/relic/mixed/newratiogen/
offline_compute=false
num_components=1

{
for i in {2..2};
do
  echo Processing Workflow $i
  indir=$basedir/$i/combined/artifacts/
  outdir=$basedir/$i/combined/inferred/

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
                          --nb_name=combined_all \
                          --out=$outdir \
                          --pre_compute=True \
                          --inter_contain=0.1 \
                          --intra_contain=0.1 \
                          --inter_cell=0.1 \
                          --intra_cell=0.1 \
                          --max_n_edges=$((num_artifacts-num_components)) \
                          --result_prefix=nppo_first_2

  echo "$i; `date` ;Completed"

done
} 2>&1 | tee -a $basedir/relic_run.log