#!/bin/bash

basedir=/tank/local/suhail/data/relic/mixed/grid/
outdir=/tank/local/suhail/data/relic/
setname=gridmixed

tar -cf $outdir$setname.tar -T /dev/null

for i in {1..16}
do
  echo Copying Workflow $i

  tar --append -vf $outdir$setname.tar $basedir$i/combined/combined_gt_fixed.pkl --strip-components=7
  tar --append -vf $outdir$setname.tar $basedir$i/combined/inferred/*inferred_graph.csv --strip-components=7
done