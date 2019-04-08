#!/bin/sh

echo "-- launch the experiment via condor"
python -m memory_profiler nddetection.py $@
