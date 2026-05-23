#!/bin/bash

docker build -t mlagentbench-thesis .

docker rm -f mlagentbench-thesis-ctr || true

clear

docker run --gpus all --user root -w /MLAgentBench \
  --name mlagentbench-thesis-ctr \
  -p 8001:8000 -p 8002:8002 \
  -v ${PWD}:/MLAgentBench -it mlagentbench-thesis
