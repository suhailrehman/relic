#!/bin/bash

basedir=/tank/local/suhail/data/relic/mixed/mixed_sigmod_flat/
offline_compute=true
num_components=1
combined='combined'

{
#for i in 2 3 4;
for f in "$basedir"*;
do
  if [ -d ${f} ]; then
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
      echo "$i; `date` ; Computing Baseline"
      ./offline.sh -i $indir -o $outdir -f baseline -n 2
    fi

    num_artifacts=`ls -1q $indir/*.csv | wc -l`
    echo "Number of Artifacts:$num_artifacts"
    echo "Number of Edges to Be inferred:" $((num_artifacts-num_components))

    echo "$i; `date` ; Running RELIC"

    # Final Inference Job
#    python ../relic/core.py --artifact_dir=$indir \
#                            --nb_name=$i \
#                            --out=$outdir \
#                            --pre_compute=True \
#                            --pre_cluster=False \
#                            --celljaccard=True \
#                            --cellcontain=False \
#                            --join=False \
#                            --groupby=False \
#                            --pivot=False \
#                            --inter_contain=0.0 \
#                            --intra_contain=0.0 \
#                            --inter_cell=0.0 \
#                            --intra_cell=0.0 \
#                            --max_n_edges=$((num_artifacts-num_components)) \
#                            --result_prefix=cell_


#      python ../relic/core.py --artifact_dir=$indir \
#                            --nb_name=$i \
#                            --out=$outdir \
#                            --pre_compute=True \
#                            --pre_cluster=False \
#                            --celljaccard=True \
#                            --cellcontain=False \
#                            --join=True \
#                            --groupby=True \
#                            --pivot=True \
#                            --inter_contain=0.0 \
#                            --intra_contain=0.0 \
#                            --inter_cell=0.0 \
#                            --intra_cell=0.0 \
#                            --max_n_edges=$((num_artifacts-num_components)) \
#                            --result_prefix=cell+detectors_

        python ../relic/core.py --artifact_dir=$indir \
                            --nb_name=$i \
                            --out=$outdir \
                            --pre_compute=True \
                            --inter_contain=0.99 \
                            --intra_contain=0.99 \
                            --inter_cell=0.1 \
                            --intra_cell=0.1 \
                            --max_n_edges=$((num_artifacts-num_components)) \
                            --result_prefix=relic_ &

        python ../relic/core.py --artifact_dir=$indir \
                                  --nb_name=$i \
                                  --out=$outdir \
                                  --pre_compute=True \
                                  --max_n_edges=$((num_artifacts-num_components)) \
                                  --result_prefix=baseline_ \
                                  --baseline=True

    echo "$i; `date` ;Completed"
  fi

done
} 2>&1 | tee -a $basedir/relic_run.log

echo "Relic run completed at $basedir, Time: `date`" | mailx -s "RELIC Lineage Inference System" suhail@uchicago.edu
