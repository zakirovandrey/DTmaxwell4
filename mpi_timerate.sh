#!/bin/bash
for n in `seq 0 63`; do \
echo -n "$n	";
cat $1 |grep "node $n" |tail -n +$(($n+5)) |cut -d" " -f 6 |awk '{ total += $1; count++ } END { print total/count }'; 
done
