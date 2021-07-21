#!/bin/bash

for i in {1..11}
  do
    arg=$((i*2))
    echo Running with fib arg: $arg
    make clean && make -j exsum DELAY=$arg
    ./exsum --gtest_filter=PerfTest.Double
  done

