indir=/tank/local/suhail/data/relic/single/1/artifacts/
outdir=/tank/local/suhail/data/relic/single/1/inferred/


python ../relic/core.py --artifact_dir=$indir \
                    --nb_name=combined_all \
                    --out=$outdir \
                    --pre_compute=False \
                    --inter_contain=0.1 \
                    --intra_contain=0.1 \
                    --inter_cell=0.1 \
                    --intra_cell=0.1 \
                    --result_prefix=relic_


#python ../relic/core.py --artifact_dir=$indir \
#                        --nb_name=combined_all \
#                        --out=$outdir \
#                        --pre_compute=False \
#                        --pre_cluster=False \
#                        --celljaccard=True \
#                        --cellcontain=False \
#                        --join=False \
#                        --groupby=False \
#                        --pivot=False \
#                        --inter_contain=0.0 \
#                        --intra_contain=0.0 \
#                        --inter_cell=0.0 \
#                        --intra_cell=0.0 \
#                        --result_prefix=cell_