#!/bin/bash

#basedir=/tank/local/suhail/data/relic/single/
#basedir=/tank/local/suhail/data/relic/mixed/mixed_sigmod_new/
#basedir=/tank/local/suhail/data/relic/relic_datasets_vldb_2021/real-world_sample_timing/
basedir=/tank/local/suhail/data/relic/relic_datasets_vldb_2021/dataset_flat_exact_sample_timing/
offline_compute=false
num_components=1
combined=''
threads=5

samples=(0.05 0.10 0.25 0.5 0.75)
sample_index_flags=('False' 'True')



#samples=(0.05)
#sample_index_flags=('True')

{
#for i in {1..16};
for f in "$basedir"*;
#for i in  10 11 12
do
  if [ -d ${f} ]; then
    thread_counter=0
    for sampleFrac in ${samples[@]}; do
      for sampleIndex in ${sample_index_flags[@]}; do
        i=$(basename $f)
        echo Processing Workflow $i
        indir=$basedir/$i/$combined/artifacts/
        outdir=$basedir/$i/$combined/inferred/sample_${sampleIndex}_${sampleFrac}

        mkdir -p $outdir
        if $offline_compute;
        then
            echo "$i; `date` ; Computing PPO with Sample Fraction $sampleFrac"
            ./offline_sample.sh -i $indir -o $outdir -f ppo -n 2 -t $threads -s $sampleFrac -I $sampleIndex
            echo "$i; `date` ; Computing Joins with Sample Fraction $sampleFrac"
            ./offline_sample.sh -i $indir -o $outdir -f join -n 3 -t $threads -s $sampleFrac -I $sampleIndex
             echo "$i; `date` ; Computing Groupbys with Sample Fraction $sampleFrac"
            ./offline_sample.sh -i $indir -o $outdir -f groupby -n 2 -t $threads -s $sampleFrac -I $sampleIndex
            echo "$i; `date` ; Computing Pivots with Sample Fraction $sampleFrac"
            ./offline_sample.sh -i $indir -o $outdir -f pivot -n 2 -t $threads -s $sampleFrac -I $sampleIndex
        fi

        num_artifacts=`ls -1q $indir/*.csv | wc -l`
        echo "Number of Artifacts:$num_artifacts"
        echo "Number of Edges to Be inferred:" $((num_artifacts-num_components))

        echo "$i; `date` ; Running RELIC"

        # Final Inference Job
        python ../relic/core.py --artifact_dir=$indir \
                                --nb_name=$i \
                                --out=$outdir \
                                --pre_compute=False \
                                --inter_contain=0.99 \
                                --intra_contain=0.99 \
                                --inter_cell=0.1 \
                                --intra_cell=0.1 \
                                --max_n_edges=$((num_artifacts-num_components)) \
                                --result_prefix=relic_ \
                                --sample_frac=$sampleFrac \
                                --sample_index=$sampleIndex \
                                --store_scores &
        pids[${thread_counter}]=$!
        let "thread_counter+=1"

        echo "$i; `date` ;Completed"
      done
    done

    # Wait until all the procs for this workflow are done
    for pid in ${pids[*]}; do
        wait $pid
    done

  fi
done
} 2>&1 | tee -a $basedir/relic_run.log

echo "Relic run completed at $basedir, Time: `date`" | mailx -s "RELIC Lineage Inference System" suhail@uchicago.edu
