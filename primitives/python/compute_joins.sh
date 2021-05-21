#input_spec='/tank/local/suhail/data/relic/mixed/syn2/inferred/joins/'
#output_dir='/tank/local/suhail/data/relic/mixed/syn2/inferred/join_score/'

input_dir='/tank/local/suhail/data/relic/mixed/syn2/inferred/join_score/'
outfile='/tank/local/suhail/data/relic/mixed/syn2/inferred/triple_dict.pkl'

python compute_joins.py $input_dir $outfile

#for f in "$input_spec"*; do
#  #echo $f
#  mkdir -p $output_dir
#  filename=`basename $f`
#  #echo $output_dir$filename
#  python compute_joins.py $f $output_dir$filename &
#done
#
#wait