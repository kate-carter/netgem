#correlates the subtype vectors (columns) against each other for each
# concept sheet in the round2networks-2026.xlsx workbook

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

# List of concepts to process (each is a sheet name in the workbook)
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

# Base paths
input_path = '/Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory_human_10/round2networks-2026.xlsx'
output_dir = '/Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory_human_10/mahalanobis/'

# The subtype table on each sheet starts with a header row of subtype names
# (Excel row 281), preceded by two index columns (a per-category counter and
# the feature word) on the far left. skiprows counts the rows above that
# header (0-indexed), so header=0 lands on the subtype-name row itself.
header_skiprows = 280


start_time = time.time()
# set up argument parser for method
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='mahalanobis',
                    help='Method to use for correlation: pearson, spearman, mahalanobis')
args = parser.parse_args()
method = args.method

results = []

#loop to process each concept (sheet)
for concept in concept_name:
        file_start = time.time()
        print(f"Processing {concept}")
        try:
            # Load the subtype table for this concept: two index columns on
            # the far left, subtype names as the header row, correlate
            # column×column to create an N_subtypes×N_subtypes matrix
            matrix_emb = pd.read_excel(
                input_path, sheet_name=concept,
                skiprows=header_skiprows, header=0, index_col=[0, 1]
            )
            # Drop trailing unnamed/blank columns that aren't real subtypes
            matrix_emb = matrix_emb.loc[:, ~matrix_emb.columns.astype(str).str.startswith('Unnamed')]
            matrix_emb = matrix_emb.dropna(how='all')
            print(f"  Loaded shape: {matrix_emb.shape}")
            X = matrix_emb.values.T  # transpose so each subtype (column) becomes a vector

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


                # exponential kernel similarity conversion (Euclidean/Laplacian form)
                #sigma = np.median(off_diag_distances)
                #sigma = max(sigma, 1e-12)  # guard
                #matrix_corr = np.exp(-d / sigma)
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
            
            # Save matrix with consistent naming: concept_subtypes_method.csv
            # Rows and columns are both labeled with the subtype names, since
            # the matrix is subtype x subtype (symmetric).
            subtype_labels = matrix_emb.columns
            output_file = f'{output_dir}{concept}_subtypes_{method}.csv'
            pd.DataFrame(matrix_corr, index=subtype_labels, columns=subtype_labels).to_csv(
                output_file, header=True, index=True
            )

            elapsed = time.time() - file_start
            print(f"  ✓ Saved: {concept}_subtypes_{method}.csv ({elapsed:.2f}s)")
            results.append({
                'concept': concept,
                'method': method,
                'matrix_shape': matrix_corr.shape
            })
        except Exception as e:
            logging.error(f"ERROR processing {concept}: {e}")
            continue

total_time = time.time() - start_time
print("\n" + "=" * 60)
print(f"Completed! Processed {len(concept_name)} concepts, elapsed time of {total_time / 60:.2f} minutes")
print(f"Output directory: {output_dir}")