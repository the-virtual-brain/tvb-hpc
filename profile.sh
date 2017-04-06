#!/bin/bash

# blockdim=${1:-32}
# griddim=${2:-32}
# outfile=${3:-"./profile.out"}
# n_time=${4:-"400"}

outfile=${1:-"./profile.out"}
shift 1
parsweep_opts="$@"

metrics=${METRICS:-"gld_throughput,gld_efficiency,achieved_occupancy,single_precision_fu_utilization"}

echo "metrics used will be $metrics (METRICS=x,y,z ./profile.sh ... to change)"
opts=" -p $parsweep_opts "
echo "options to ./parsweep.py are $opts "

sleep 2

set -o verbose
set -eu

./parsweep.py $opts > $outfile 2>&1

nvprof ./parsweep.py $opts >> $outfile 2>&1

nvprof --metrics $metrics ./parsweep.py $opts >> $outfile 2>&1
