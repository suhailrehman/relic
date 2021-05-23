artifact_dir='/tank/local/suhail/data/relic/mixed/syn2/artifacts/'
output_file='/tank/local/suhail/data/relic/mixed/syn2/inferred/groupbys/gb_pairs.txt'
input_spec='/tank/local/suhail/data/relic/mixed/syn2/inferred/groupbys/'
output_dir='/tank/local/suhail/data/relic/mixed/syn2/inferred/groupby_score/'

#Generate Pairs
#python compute_tuple_distances.py $artifact_dir $output_file

#split pairs

#split -da 2 -l $((`wc -l < gb_pairs.txt` / 20)) gb_pairs.txt gb_pairs_part_ --additional-suffix=".txt"

for f in "$input_spec"*; do
  #echo $f
  mkdir -p $output_dir
  filename=`basename $f`
  #echo $output_dir$filename
  python compute_nppos.py $f $output_dir$filename $artifact_dir &
done

wait