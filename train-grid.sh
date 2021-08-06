#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python3 embed.py \
       -dim 2 \
       -lr 0.15 \
       -epochs 700 \
       -negs 10 \
       -burnin 20 \
       -ndproc 4 \
       -model distance \
       -manifold poincare \
       -dset robot_planning/grid_nodes9.csv \
       -checkpoint grid_nodes9.pth \
       -batchsize 20 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 2