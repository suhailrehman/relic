indir=/home/suhail/Data/relic/mixed_sigmod_flat/1/combined/artifacts/
outdir=/home/suhail/Data/relic/mixed_sigmod_flat/1/combined/inferred/lsh_0.4/
graph_file=/mnt/c/Users/suhai/Git/relic-datalake/notebooks/results/synthetic/1/0.40/connectivity_graph.gpkl


python ../relic/core.py --artifact_dir=$indir \
                    --nb_name=combined_all \
                    --out=$outdir \
                    --pre_compute=False \
                    --inter_contain=0.1 \
                    --intra_contain=0.1 \
                    --inter_cell=0.1 \
                    --intra_cell=0.1 \
                    --result_prefix=relic_lsh_0.4 \
                    --match_schema=False \
                    --lsh_graph_file=$graph_file
