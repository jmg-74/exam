#!/bin/bash
for BS in 2 3 4 5 6 7
do
    ./mnist_mem_monitor.py --batch-size $BS --disable-dp
    ./mnist_mem_monitor.py --batch-size $BS 2>/dev/null
done
