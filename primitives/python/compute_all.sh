artifact_dir='/tank/local/suhail/data/relic/mixed/syn100/artifacts/'
output_dir='/tank/local/suhail/data/relic/mixed/syn100/inferred/'

# Number of processes to spawn for distance computations
num_procs=35

distfunc='join' # cellcontainment

# Create inferred dir if it does not exist:
mkdir -p $output_dir

# Generate tuples first
#python compute_tuple_distances.py --mode=enumerate --input=$artifact_dir --output=$output_dir/tuples.csv
python compute_tuple_distances.py --mode=enumerate --input=$artifact_dir --output=$output_dir/triples.csv --ntuples=3 --func=$distfunc


#mkdir -p $output_dir/tuples
mkdir -p $output_dir/triples


# Split tuples into $num_procs files to distribute
#split -da 2 -l $((`wc -l < $output_dir/tuples.csv` / 20)) $output_dir/tuples.csv $output_dir/tuples/tuples_part_ --additional-suffix=".csv"
split -da 2 -l $((`wc -l < $output_dir/triples.csv` / 20)) $output_dir/triples.csv $output_dir/triples/triples_part_ --additional-suffix=".csv"


#Compute cell-jacard
mkdir -p $output_dir/$distfunc

#for f in "$output_dir/tuples/"*; do
for f in "$output_dir/triples/"*; do
  filename=`basename $f`
  python compute_tuple_distances.py --mode=compute --slice=$f  --input=$artifact_dir --output=$output_dir/$distfunc/$filename --func=$distfunc&
done

wait

# combine computed scores
#python compute_tuple_distances.py --mode=combine  --input=$output_dir/$distfunc/$filename --output=$output_dir/$distfunc.pkl

# Single distance test
#python compute_tuple_distances.py --mode=compute --slice=$output_dir/tuples/tuples_part_00.csv  --input=$artifact_dir --output=$output_dir/celljaccard/tuples_part_00.csv
