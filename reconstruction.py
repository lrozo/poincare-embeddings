#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hype.graph import eval_reconstruction, load_adjacency_matrix
import argparse
import numpy as np
import torch
import os
import timeit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from hype import MANIFOLDS, MODELS

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to checkpoint', default='grid_nodes.pth.best')
parser.add_argument('-workers', default=1, type=int, help='Number of workers')
parser.add_argument('-sample', type=int, help='Sample size')
parser.add_argument('-quiet', action='store_true', default=False)
args = parser.parse_args()

chkpnt = torch.load(args.file)
dset = chkpnt['conf']['dset']
if not os.path.exists(dset):
    raise ValueError("Can't find dset!")

format = 'hdf5' if dset.endswith('.h5') else 'csv'
dset = load_adjacency_matrix(dset, format, objects=chkpnt['objects'])

colors = cm.rainbow(np.linspace(0, 1, 25))
#circle = visualization.PoincareDisk(point_type='ball')
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
# circle.add_points(gs.array([[0, 0]]))
#circle.set_ax(ax)
#circle.draw(ax=ax)
embeddings = chkpnt['embeddings'].numpy()
for i_embedding, embedding in enumerate(embeddings):
    plt.scatter(embedding[0], embedding[1], alpha=1.0, color=colors[i_embedding],
                label=i_embedding)
    plt.annotate(chkpnt['objects'][i_embedding], (embedding[0], embedding[1]))
# plt.legend()
plt.show()

sample_size = args.sample or len(dset['ids'])
sample = np.random.choice(len(dset['ids']), size=sample_size, replace=False)

adj = {}

for i in sample:
    end = dset['offsets'][i + 1] if i + 1 < len(dset['offsets']) \
        else len(dset['neighbors'])
    adj[dset['ids'][i]] = set(dset['neighbors'][dset['offsets'][i]:end])
manifold = MANIFOLDS[chkpnt['conf']['manifold']]()

manifold = MANIFOLDS[chkpnt['conf']['manifold']]()
model = MODELS[chkpnt['conf']['model']](
    manifold,
    dim=chkpnt['conf']['dim'],
    size=chkpnt['embeddings'].size(0),
    sparse=chkpnt['conf']['sparse']
)
model.load_state_dict(chkpnt['model'])

lt = chkpnt['embeddings']
if not isinstance(lt, torch.Tensor):
    lt = torch.from_numpy(lt).cuda()



tstart = timeit.default_timer()
meanrank, maprank = eval_reconstruction(adj, model, workers=args.workers,
    progress=not args.quiet)
etime = timeit.default_timer() - tstart

print(f'Mean rank: {meanrank}, mAP rank: {maprank}, time: {etime}')
