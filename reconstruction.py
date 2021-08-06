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


def plot_geodesics(embeddings_array, ref_id, target_id, geo_color, geodesics_steps):
    tangent_vec = manifold.log(embeddings_array[ref_id, :], embeddings[target_id, :])
    t = np.linspace(0.01, 1, geodesics_steps)
    geodesics = [embeddings[ref_id, :]]
    for step in t:
        geodesics.append(manifold.exp(embeddings[ref_id, :], step * tangent_vec))
    geodesics_array = np.array(geodesics)
    plt.plot(geodesics_array[:, 0], geodesics_array[:, 1], color=geo_color)


np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to checkpoint', default='grid_nodes9.pth.best')
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

colors = cm.rainbow(np.linspace(0, 1, 81))
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
circle = plt.Circle((0, 0), radius=1., color='black', fill=False)
ax.add_artist(circle)
embeddings = chkpnt['embeddings'].numpy()
for i_embedding, embedding in enumerate(embeddings):
    node = chkpnt['objects'][i_embedding]
    plt.scatter(embedding[0], embedding[1], alpha=1.0, color=colors[node])
    plt.annotate(node, (embedding[0], embedding[1]))

targets = np.array([1, 6, 14, 20, 21, 34, 35, 48])
for target in targets:
    plot_geodesics(embeddings, ref_id=0, target_id=target, geo_color=colors[target], geodesics_steps=30)
plt.show()

