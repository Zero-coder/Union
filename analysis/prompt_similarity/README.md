# Figure 5: prompt representation structure analysis

This directory contains the numerical similarity matrix, domain labels, results,
and code for reproducing the quantitative calculations associated with Figure 5.
The analysis covers 16 datasets in nine predefined domains.

## Data source and scope

The author confirms that the supplied matrix matches the model-exported
similarity values entry by entry. This confirmation covers the complete
16-by-16 matrix, rather than only the rounded summary metrics.
The preparation history, author confirmation, and file hashes are recorded
in [`provenance.json`](provenance.json). Original prompt embeddings and
training checkpoints are not included in this analysis package.

The domain assignments reproduce the labels displayed in Figure 5. They are
predefined groups, not labels fitted by a clustering algorithm. The scripts
reproduce calculations from these supplied inputs, not model training.

## Files

| File | Contents |
| --- | --- |
| `similarity_matrix.csv` | Symmetric 16-by-16 matrix, with dataset names in the header and first column. |
| `dataset_domains.csv` | Dataset-to-domain assignments used for Figure 5. |
| `metrics.json` | Recomputed metrics, pair counts, domain sizes, per-dataset scores, and input hashes. |
| `provenance.json` | Matrix preparation history and author confirmation. |
| `evaluate_prompt_structure.py` | Python implementation of the quantitative analysis. |
| `test_evaluate_prompt_structure.py` | Analytical unit tests and input-validation tests. |

## Reproduce the calculations

Python 3.8 or newer is sufficient; no third-party packages are required.
From the repository root, run:

```bash
python analysis/prompt_similarity/evaluate_prompt_structure.py --matrix analysis/prompt_similarity/similarity_matrix.csv --domains analysis/prompt_similarity/dataset_domains.csv --output analysis/prompt_similarity/metrics.json
```

The evaluator computes all results from its inputs; it does not contain the
reported metric values. Run the analytical tests with:

```bash
python -m unittest discover -s analysis/prompt_similarity -p "test_*.py" -v
```

## Definitions and results

| Metric | Value (three decimal places) |
| --- | ---: |
| Mean within-domain cosine similarity | 0.479 |
| Mean cross-domain cosine similarity | 0.296 |
| Cosine-distance silhouette score | 0.027 |

For similarity matrix `S`, define cosine dissimilarity as `D[i,j] = 1 - S[i,j]`.
The pairwise means use unique unordered off-diagonal pairs (`i < j`): 11
within-domain pairs and 109 cross-domain pairs. Every eligible pair receives
equal weight; domains are not reweighted to equal size.

For each dataset in a non-singleton domain, `a` is its mean distance to the other
members of its domain and `b` is the smallest mean distance to any other domain.
The silhouette value is `(b - a) / max(a, b)`. Singleton domains and a zero
denominator receive a score of zero. The overall score is the unweighted mean
across all 16 datasets, including the five singleton domains.

The higher within-domain mean indicates greater average similarity within the
predefined groups. The near-zero silhouette indicates weak overall separation
under these labels. These metrics characterize representation structure;
adaptation performance requires a separate evaluation on the target task.

## Input validation and output

The matrix CSV must have header `dataset,<dataset names>`, matching row labels,
finite entries in `[-1, 1]`, symmetry, and a unit diagonal. The domains CSV must
have the exact header `dataset,domain`, with one assignment per dataset.
Domain rows may appear in any order. Missing, extra, empty, or duplicate labels
are rejected. Symmetry and diagonal checks use absolute tolerance `1e-10`.
Silhouette evaluation requires at least two domains and fewer domains than
datasets. The evaluator does not silently repair inputs or check positive
semidefiniteness or the source of the representations.

The output JSON includes full-precision calculations and three-decimal display
values, calculation conventions, per-dataset scores, and SHA-256 hashes of the
inputs. Only input filenames are recorded, not absolute paths.
