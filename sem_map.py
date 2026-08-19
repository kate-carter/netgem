#semantic map: visualize each concept's subtype similarity matrix as a 2D
# map of points, where proximity between two points reflects how similar
# those two subtypes are (closer together = more similar, farther apart =
# less similar). Reads the labeled subtype x subtype matrices already
# produced by corr_human.py. All concepts are drawn as subplots in a single
# combined output file, with labels jittered/decluttered so they don't stack
# on top of each other in dense clusters.

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe backend for saving figures
from matplotlib import pyplot as plt
import argparse

# List of concepts to process (must match the *_subtypes_<method>.csv files
# produced by corr_human.py)
concept_name = [
    'KEY',
    'PUMPKIN',
    'GRASS',
    'COOKIE',
    'PICKLE',
    'PILLOW',
    'KNIFE',
    'WOOD',
    'PHONE',
    'CAR'
]

# Base paths: read the labeled correlation matrices corr_human.py wrote here,
# and save the resulting combined map back into the same folder
data_dir = '/Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory_human_10/mahalanobis/'
output_dir = '/Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory_human_10/mahalanobis/'

# set up argument parser for method (matches corr_human.py's --method values)
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='mahalanobis',
                    help='Which correlation matrix to map: pearson, spearman, mahalanobis, cosine')
args = parser.parse_args()
method = args.method


def classical_mds(distance_matrix, n_components=2):
    """Classical (Torgerson) multidimensional scaling.

    Takes a symmetric distance/dissimilarity matrix and returns coordinates
    in n_components dimensions such that Euclidean distance between points
    approximates the input distances as closely as possible, i.e. subtypes
    that are more similar end up plotted closer together.
    """
    n = distance_matrix.shape[0]
    D2 = distance_matrix ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    # sort by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # keep only positive eigenvalues (guards against numerical noise)
    positive = eigvals > 1e-10
    k = min(n_components, int(positive.sum()))
    coords = eigvecs[:, :k] * np.sqrt(eigvals[:k])

    # pad with zeros if fewer than n_components valid dimensions were found
    if coords.shape[1] < n_components:
        pad = np.zeros((n, n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])

    return coords


def declutter_labels(coords, seed=0, iterations=400, min_dist_frac=0.055,
                      own_dist_frac=0.026, step=0.6):
    """Jitter/repel label positions so they don't overlap each other or sit
    directly on top of their own point.

    Starts each label offset from its point at a small standoff radius
    (spread around the circle via the golden angle so nearby points don't
    all start in the same direction), then iteratively: (1) pulls any label
    that drifted back within own_dist of its own point back out, and
    (2) pushes apart any pair of labels closer than min_dist. Distances are
    fractions of the overall coordinate span. Returns an array of label
    positions, separate from the (unmoved) point coordinates.
    """
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span == 0] = 1.0
    scale = np.linalg.norm(span)
    min_dist = min_dist_frac * scale
    own_dist = own_dist_frac * scale

    golden_angle = np.pi * (3 - np.sqrt(5))
    angles = golden_angle * np.arange(n) + rng.uniform(0, 0.3, size=n)
    init_radius = own_dist + rng.uniform(0, own_dist * 0.5, size=n)
    offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * init_radius[:, None]
    label_pos = coords + offsets

    for _ in range(iterations):
        moved = False

        # keep every label at least own_dist away from its own point
        for i in range(n):
            delta = label_pos[i] - coords[i]
            dist = np.linalg.norm(delta)
            if dist < own_dist:
                moved = True
                if dist < 1e-9:
                    delta = rng.uniform(-1, 1, size=2)
                    dist = np.linalg.norm(delta)
                label_pos[i] = coords[i] + delta / dist * own_dist

        # push apart any pair of labels that are too close to each other
        for i in range(n):
            for j in range(i + 1, n):
                delta = label_pos[i] - label_pos[j]
                dist = np.linalg.norm(delta)
                if dist < min_dist:
                    moved = True
                    if dist < 1e-9:
                        delta = rng.uniform(-1, 1, size=2)
                        dist = np.linalg.norm(delta)
                    push = (min_dist - dist) / 2 * step * (delta / dist)
                    label_pos[i] += push
                    label_pos[j] -= push

        if not moved:
            break

    return label_pos


results = []
maps = []

#loop to process each concept and compute its map (drawing happens after,
# once all concepts have been laid out into a shared grid of subplots)
for concept in concept_name:
    input_file = f'{data_dir}{concept}_subtypes_{method}.csv'
    print(f"Processing {concept}")
    try:
        # Load the labeled subtype x subtype similarity matrix
        matrix_corr = pd.read_csv(input_file, index_col=0)
        labels = matrix_corr.index.tolist()
        sim = matrix_corr.values
        print(f"  Loaded shape: {sim.shape}")

        # Convert similarity to distance so proximity = similarity (more
        # similar subtypes -> smaller distance). The diagonal was zeroed out
        # when the similarity matrix was saved (rather than left at 1 for
        # self-similarity), so force it back to a true self-distance of 0
        # after the conversion.
        dist = 1 - sim
        np.fill_diagonal(dist, 0)

        coords = classical_mds(dist, n_components=2)
        label_pos = declutter_labels(coords)

        maps.append({
            'concept': concept,
            'labels': labels,
            'coords': coords,
            'label_pos': label_pos
        })
        results.append({
            'concept': concept,
            'method': method,
            'n_subtypes': len(labels)
        })
    except Exception as e:
        print(f"ERROR processing {concept}: {e}")
        continue

# Draw every concept's map as a subplot in one combined figure/file
n_maps = len(maps)
ncols = min(5, n_maps) if n_maps else 1
nrows = math.ceil(n_maps / ncols) if n_maps else 1
fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)
axes_flat = axes.flatten()

for ax, m in zip(axes_flat, maps):
    coords = m['coords']
    label_pos = m['label_pos']
    ax.scatter(coords[:, 0], coords[:, 1], s=60, color='steelblue', edgecolor='black', zorder=3)
    for (px, py), (lx, ly), label in zip(coords, label_pos, m['labels']):
        # thin leader line if the label had to be pushed away from its point
        if math.hypot(lx - px, ly - py) > 1e-9:
            ax.plot([px, lx], [py, ly], color='gray', linewidth=0.4, zorder=2)
        ax.annotate(label, (lx, ly), fontsize=6.5, ha='center', va='center', zorder=4)
    ax.set_title(f"{m['concept']} ({method})", fontsize=11)
    ax.axhline(0, color='lightgray', linewidth=0.5, zorder=1)
    ax.axvline(0, color='lightgray', linewidth=0.5, zorder=1)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([])
    ax.set_yticks([])

# hide any unused grid cells (when n_maps doesn't fill the grid exactly)
for ax in axes_flat[n_maps:]:
    ax.axis('off')

fig.suptitle(f'Subtype Semantic Maps ({method})', fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.97))

output_file = f'{output_dir}all_concepts_semmap_{method}.png'
fig.savefig(output_file, dpi=200)
plt.close(fig)

print("\n" + "=" * 60)
print(f"Completed! Processed {len(results)}/{len(concept_name)} concepts")
print(f"✓ Saved combined map: all_concepts_semmap_{method}.png")
print(f"Output directory: {output_dir}")
