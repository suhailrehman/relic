
indir=/tank/local/suhail/data/relic/newgen/combined_all/artifacts/
outdir=/tank/local/suhail/data/relic/newgen/combined_all/inferred/

#mkdir -p $outdir
#./offline.sh -i $indir -o $outdir -f ppo -n 2
#./offline.sh -i $indir -o $outdir -f join -n 3
#./offline.sh -i $indir -o $outdir -f groupby -n 2
#./offline.sh -i $indir -o $outdir -f pivot -n 2

# Comvine Only:
#for distfunc in ppo join groupby pivot;
#do
#echo $distfunc
#python ../relic/offline.py --mode=combine  --input=$outdir/$distfunc/ --output=$outdir/$distfunc.csv
#done

# Final Inference Job
python ../relic/core.py --artifact_dir=$indir \
                        --nb_name=combined_all \
                        --out=$outdir \
                        --pre_compute=True