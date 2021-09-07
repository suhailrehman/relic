#!/bin/bash

basedir=/tank/local/suhail/data/relic/mixed/grid/
offline_compute=true
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
    echo "$i; `date` ; Computing Baseline"
    ./offline.sh -i $indir -o $outdir -f baseline -n 2
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
                          --max_n_edges=$((num_artifacts-num_components)) \
                          --result_prefix=baseline_
                          --baseline=True

  echo "$i; `date` ;Completed"

done
} 2>&1 | tee -a $basedir/relic_run.log