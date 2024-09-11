#!/usr/bin/bash

cwd_realpath=$(realpath ./venv)
source ./venv/bin/activate
parallel -N0 python3 ./sweep.py --sweep-id $2 ::: $(seq 1 $1)
wait
