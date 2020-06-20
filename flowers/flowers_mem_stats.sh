#!/bin/bash
for HU in 32 64 512  # seq 32 8 128
do
  for BS in 2 3 4 5 6 7
  do
    for F in 1 2 4
    do
      ./flowers_mem_monitor.py --hidden-units $HU --batch-size $BS -f $F --disable-dp
      ./flowers_mem_monitor.py --hidden-units $HU --batch-size $BS -f $F
    done
  done
done
