# Prompt representation structure analysis

This directory provides code for evaluating domain structure from a labeled
dataset-level cosine-similarity matrix. It computes within-domain similarity,
cross-domain similarity, and the cosine-distance silhouette score.

The release contains the evaluator and analytical unit tests. Similarity
matrices, prompt embeddings, model checkpoints, and experimental results are
not included in this code release; users supply their own input files.
Test fixtures are used only to check the calculations.

## Requirements

Python 3.8 or newer. The evaluator uses only the Python standard library.

## Inputs

- `--matrix`: UTF-8 CSV with header `dataset,<dataset names>`. Each subsequent
  row contains the dataset name and its similarities in the same column order.
  The matrix must be square and symmetric, have a unit diagonal, and contain
  finite entries in `[-1, 1]`.
- `--domains`: UTF-8 CSV with the exact header `dataset,domain`, containing
  one assignment for every dataset in the matrix. Row order is unrestricted.
- `--output`: destination JSON file, defaulting to `metrics.json`.

Use a matrix computed from the dataset-level representations and preserve the
data-preparation and experiment settings alongside the inputs. Validation in
this utility checks matrix structure and label alignment; it does not check
positive semidefiniteness or the source of the representations.

## Usage

From the repository root, supply the paths to your input files:

```bash
python analysis/prompt_similarity/evaluate_prompt_structure.py --matrix similarity_matrix.csv --domains dataset_domains.csv --output metrics.json
```

Run the analytical tests:

```bash
python -m unittest discover -s analysis/prompt_similarity -p "test_*.py" -v
```

## Metric definitions

For similarity matrix `S`, define cosine dissimilarity as `D[i,j] = 1 - S[i,j]`.

- **Mean within-domain similarity:** arithmetic mean over unique unordered
  dataset pairs `i < j` with the same domain label.
- **Mean cross-domain similarity:** arithmetic mean over unique unordered
  dataset pairs `i < j` with different domain labels.
- **Silhouette score:** for dataset `i`, let `a` be its mean distance to the
  other members of its domain and `b` the smallest mean distance to any other
  domain. Its score is `(b - a) / max(a, b)`. Singleton domains and a zero
  denominator receive a score of zero. The overall score is the unweighted
  mean over all datasets, including singleton domains.

Diagonal entries are excluded from both pairwise means. Each eligible pair
receives equal weight; domains are not reweighted to equal size. Silhouette
evaluation requires at least two domains and fewer domains than datasets.
The supplied domain labels are treated as predefined groups.

## Output

The JSON contains full-precision and three-decimal metric values, domain sizes,
pair counts, per-dataset silhouette scores, calculation conventions, and SHA-256
hashes of both input files. Only input filenames are recorded, not absolute paths.

These metrics characterize the organization of the supplied representations.
Adaptation performance requires a separate evaluation on the target task.
