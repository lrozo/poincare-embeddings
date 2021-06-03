#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python3 embed.py \
       -dim 2 \
       -lr 0.3 \
       -epochs 300 \
       -negs 5 \
       -burnin 20 \
       -ndproc 4 \
       -model distance \
       -manifold poincare \
       -dset robot_planning/grid_nodes.csv \
       -checkpoint grid_nodes.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 2