#!/bin/bash
for BS in 2 3 4 5 6 7 
do
    ./mem_mnist.py --batch-size $BS --disable-dp
    ./mem_mnist.py --batch-size $BS 2>/dev/null
done
