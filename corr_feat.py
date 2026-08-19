#correlates the concept vectors (columns) in feature-strengths.xlsx against
# each other, based on each concept's strength profile across all features
# correlates CONCEPTS (not individual subtypes, not individual features)

# pip import packages and then..
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import time
import scipy.spatial.distance as dist
import argparse
from scipy.spatial.distance import pdist, squareform
import logging
import os

# Base paths
input_path = '/Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory/feature-strengths.xlsx'
output_dir = '/Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory/'

start_time = time.time()
# set up argument parser for method
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='cosine',
                    help='Method to use for correlation: pearson, spearman, mahalanobis')
args = parser.parse_args()
method = args.method

results = []

#single pass: this workbook has one sheet with all concepts as columns
file_start = time.time()
print("Processing feature-strengths concept correlations")
try:
    # Load the feature-by-concept table: two index columns on the far left
    # (fidx, feature), a header row of concept names (KEY, PUMPKIN, ...),
    # correlate column x column to create an N_concepts x N_concepts matrix
    matrix_emb = pd.read_excel(input_path, sheet_name='Sheet1', header=0, index_col=[0, 1])
    matrix_emb = matrix_emb.dropna(how='all')
    print(f"  Loaded shape: {matrix_emb.shape}")
    X = matrix_emb.values.T  # transpose so each concept (column) becomes a vector

    if method == 'pearson':
        # Create correlation matrix using pearson correlation
        matrix_corr = matrix_emb.corr(
            method='pearson'
        ).values
        print(f"  Correlation matrix shape: {matrix_corr.shape}")
    elif method == 'spearman':
        # Create correlation matrix using spearman rank correlation
        matrix_corr = matrix_emb.corr(
            method='spearman'
        ).values
        print(f"  Correlation matrix shape: {matrix_corr.shape}")
    elif method == 'mahalanobis':
        #Create distance matrix using mahalanobis distance
        VI = np.linalg.pinv(np.cov(X, rowvar=False))
        dists = pdist(X, metric='mahalanobis', VI=VI)
        d = squareform(dists)
        off_diag_distances = d[np.triu_indices_from(d, k=1)]

        #convert to similarity using GAUSSIAN kernel
        sigma = np.median(off_diag_distances)
        if sigma > 0:
            sim01 = np.exp(-(d ** 2) / (2 * sigma ** 2))   # [0,1]
        else:
            sim01 = np.ones_like(d)
        matrix_corr = sim01
        print(f"  Mahalanobis distance matrix shape: {matrix_corr.shape}")
    elif method == "cosine":
        # create correlation matrix using cosine similarity
        dists = pdist(X, metric='cosine')
        matrix_corr = 1 - squareform(dists)
        print(f"  Cosine similarity matrix shape: {matrix_corr.shape}")

    # Min/max from off-diagonal only (diagonal must not set the scale)
    iu = np.triu_indices_from(matrix_corr, k=1)
    off_diag = matrix_corr[iu]
    min_val = np.min(off_diag)
    max_val = np.max(off_diag)
    print(f'min/max before normalization: {min_val}, {max_val}')

    # Normalize to [-1, 1] (full matrix; diagonal set after)
    if method == 'mahalanobis':
        if max_val > min_val:
            matrix_corr = 2 * ((matrix_corr - min_val) / (max_val - min_val)) - 1
        else:
            matrix_corr = np.zeros_like(matrix_corr, dtype=float)
        iu = np.triu_indices_from(matrix_corr, k=1)
        off_diag2 = matrix_corr[iu]
        min_val2 = np.min(off_diag2)
        max_val2 = np.max(off_diag2)
        print(f'min/max after normalization: {min_val2}, {max_val2}')
    else:
        # norm to [-1, 1]
        if max_val > min_val:
            matrix_corr = 2 * ((matrix_corr - min_val) / (max_val - min_val)) - 1
        else:
            matrix_corr = np.zeros_like(matrix_corr, dtype=float)
        iu = np.triu_indices_from(matrix_corr, k=1)
        off_diag2 = matrix_corr[iu]
        min_val2 = np.min(off_diag2)
        max_val2 = np.max(off_diag2)
        print(f'min/max after normalization: {min_val2}, {max_val2}'
                        )

    # Convert diagonal from 1's to 0's
    matrix_corr = matrix_corr.copy()
    np.fill_diagonal(matrix_corr, 0)

    # Save matrix with concept names preserved as both row and column
    # labels, since the matrix is concept x concept (symmetric)
    concept_labels = matrix_emb.columns
    output_file = f'{output_dir}feature_strengths_concepts_{method}.csv'
    pd.DataFrame(matrix_corr, index=concept_labels, columns=concept_labels).to_csv(
        output_file, header=True, index=True
    )

    elapsed = time.time() - file_start
    print(f"  ✓ Saved: feature_strengths_concepts_{method}.csv ({elapsed:.2f}s)")
    results.append({
        'method': method,
        'matrix_shape': matrix_corr.shape
    })
except Exception as e:
    logging.error(f"ERROR processing feature-strengths concepts: {e}")

total_time = time.time() - start_time
print("\n" + "=" * 60)
print(f"Completed! elapsed time of {total_time / 60:.2f} minutes")
print(f"Output directory: {output_dir}")
