#!/bin/bash

# Options parsing borrowed from https://stackoverflow.com/a/29754866

# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# PIPESTATUS with a simple $?, but I don’t do that.
set -o errexit -o pipefail -o noclobber -o nounset

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=di:o:n:t:f:
LONGOPTS=debug,input:,output:,ntuples:,threads:,func:

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

d=n f=n func=ppo outDir=result inDir=- ntuples=2 threads=35
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -d|--debug)
            d=y
            shift
            ;;
        -f|--func)
            func="$2"
            shift 2
            ;;
        -o|--output)
            outDir="$2"
            shift 2
            ;;
        -i|--input)
            inDir="$2"
            shift 2
            ;;
        -n|--ntuples)
            ntuples="$2"
            shift 2
            ;;
        -t|--threads)
            threads="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

echo "Tuple Enumeration Configuration:: debug: $d, inDir: $inDir, out: $outDir, ntuples: $ntuples, threads: $threads, func: $func"



artifact_dir=$inDir
output_dir=$outDir

# Number of processes to spawn for distance computations
num_procs=$threads

distfunc=$func # cellcontainment


if [ ! -d $artifact_dir ]
then
    echo "Error: Directory $artifact_dir does not exist, exiting"
    exit 1
fi

# Create inferred dir if it does not exist:
mkdir -p $output_dir


# Generate tuples first
python ../relic/offline.py --mode=enumerate --input="$artifact_dir" --output=$output_dir/"${ntuples}_combos.csv" --ntuples=$ntuples --func=$distfunc
mkdir -p $output_dir/"${ntuples}_combos/"

# Split tuples into $num_procs files to distribute
#split -da 2 -l $((`wc -l < $output_dir/tuples.csv` / 20)) $output_dir/tuples.csv $output_dir/tuples/tuples_part_ --additional-suffix=".csv"
split -da 2 -l $((`wc -l < $output_dir/"${ntuples}_combos.csv"` / $num_procs)) $output_dir/"${ntuples}_combos.csv" $output_dir/"${ntuples}_combos"/"${ntuples}_combos"_part_ --additional-suffix=".csv"


#Compute cell-jacard
mkdir -p $output_dir/$distfunc

for f in $output_dir/"${ntuples}_combos"/*; do
  filename=`basename $f`
  python ../relic/offline.py --mode=compute --slice=$f  --input=$artifact_dir/ --output=$output_dir/$distfunc/$filename --func=$distfunc &
done

wait

# combine computed scores
python ../relic/offline.py --mode=combine  --input=$output_dir/$distfunc/ --output=$output_dir/$distfunc.csv
