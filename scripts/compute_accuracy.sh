#!/bin/bash

base_dir='/tank/local/suhail/data/relic/mixed/newgen/'

for i in {1..3};
do
relic_result=$base_dir/$i'/combined/inferred/inferred_graph.csv'
ground_truth=$base_dir/$i'/combined/combined_gt_fixed.pkl'

python3 ../relic/utils/compute_accuracy.py $relic_result $ground_truth > $base_dir/$i'/combined/inferred/relic_result.json'
head -n 4 $base_dir/$i'/combined/inferred/relic_result.json'
done