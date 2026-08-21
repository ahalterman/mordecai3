#!/bin/bash
while true; do
  n=$(pgrep -c -f "sweep.py")
  if [ "$n" = "0" ]; then break; fi
  sleep 30
done
echo "all sweeps finished"
