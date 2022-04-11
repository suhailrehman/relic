#!/bin/bash

basedir='/tank/local/suhail/data/relic/perturbed/dataset_flat_exact/0.1/0.1/0.4/20210126-181345/'
nb_name='20210126-181345'
indir="${basedir}/artifacts/"
outdir="${basedir}/inferred/"

#python ../relic/core.py --artifact_dir=$indir \
#                    --nb_name=combined_all \
#                    --g_truth_file="${basedir}/${nb_name}_gt_fixed.pkl" \
#                    --out=$outdir \
#                    --pre_compute=False \
#                    --inter_contain=0.99 \
#                    --intra_contain=0.99 \
#                    --inter_cell=0.1 \
#                    --intra_cell=0.1 \
#                    --result_prefix=smtest_ \
#                    --match_schema=True

python ../relic/core.py --artifact_dir=$indir \
                    --nb_name=combined_all \
                    --g_truth_file="${basedir}/${nb_name}_gt_fixed.pkl" \
                    --out=$outdir \
                    --pre_compute=False \
                    --inter_contain=0.99 \
                    --intra_contain=0.99 \
                    --inter_cell=0.1 \
                    --intra_cell=0.1 \
                    --result_prefix=relic_ \
                    --match_schema=False


