#!/usr/bin/env python3
"""Evaluate domain structure from a labeled cosine-similarity matrix.

Inputs are a dataset similarity matrix and dataset-to-domain assignments.
Requires Python 3.8 or newer and uses only the standard library.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


TOLERANCE = 1e-10



def _labels(values, description):
    labels = [value.strip() for value in values]
    if not labels or any(not label for label in labels):
        raise ValueError(f"{description} must contain nonempty labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{description} contains duplicate labels")
    return labels


def load_matrix(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2 or rows[0][0].strip() != "dataset":
        raise ValueError("Matrix header must start with 'dataset'")
    labels = _labels(rows[0][1:], "Matrix columns")
    n = len(labels)
    if len(rows) != n + 1:
        raise ValueError("Matrix must be square with one row per dataset")
    if any(len(row) != n + 1 for row in rows[1:]):
        raise ValueError("Every matrix row must have a dataset name and N values")
    row_labels = _labels([row[0] for row in rows[1:]], "Matrix rows")
    if row_labels != labels:
        raise ValueError("Matrix row labels must match column labels in order")
    try:
        matrix = [[float(value) for value in row[1:]] for row in rows[1:]]
    except ValueError as exc:
        raise ValueError("Matrix entries must be numeric") from exc
    validate_matrix(labels, matrix)
    return labels, matrix


def validate_matrix(labels, matrix):
    _labels(labels, "Dataset labels")
    n = len(labels)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Matrix shape must be N by N")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"Similarity at ({i}, {j}) must be finite and in [-1, 1]")
            if abs(value - matrix[j][i]) > TOLERANCE:
                raise ValueError(f"Matrix is not symmetric at ({i}, {j})")
        if abs(row[i] - 1.0) > TOLERANCE:
            raise ValueError(f"Diagonal entry for {labels[i]} must be 1")


def load_domains(path, labels):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["dataset", "domain"]:
            raise ValueError("Domain CSV header must be exactly 'dataset,domain'")
        mapping = {}
        for row in reader:
            if None in row or row["dataset"] is None or row["domain"] is None:
                raise ValueError("Each domain row must have exactly two fields")
            dataset, domain = row["dataset"].strip(), row["domain"].strip()
            if not dataset or not domain:
                raise ValueError("Dataset names and domain labels must be nonempty")
            if dataset in mapping:
                raise ValueError(f"Duplicate dataset in domain CSV: {dataset}")
            mapping[dataset] = domain
    if set(mapping) != set(labels):
        missing = sorted(set(labels) - set(mapping))
        extra = sorted(set(mapping) - set(labels))
        raise ValueError(f"Domain labels do not align: missing={missing}, extra={extra}")
    return [mapping[label] for label in labels]


def evaluate(labels, matrix, domains):
    validate_matrix(labels, matrix)
    n = len(labels)
    if len(domains) != n or any(not isinstance(x, str) or not x.strip() for x in domains):
        raise ValueError("Every dataset must have one nonempty domain label")
    groups = {}
    for i, domain in enumerate(domains):
        groups.setdefault(domain, []).append(i)
    if not 2 <= len(groups) < n:
        raise ValueError("Silhouette evaluation requires 2 <= domain count < dataset count")

    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if domains[i] == domains[j] else cross).append(matrix[i][j])
    sample_scores = []
    for i, label in enumerate(labels):
        own = groups[domains[i]]
        if len(own) == 1:
            a, b, score = None, None, 0.0
        else:
            a = math.fsum(1.0 - matrix[i][j] for j in own if j != i) / (len(own) - 1)
            b = min(
                math.fsum(1.0 - matrix[i][j] for j in members) / len(members)
                for domain, members in groups.items() if domain != domains[i]
            )
            denominator = max(a, b)
            score = (b - a) / denominator if denominator else 0.0
        sample_scores.append({
            "dataset": label,
            "domain": domains[i],
            "domain_size": len(own),
            "mean_within_domain_distance_a": a,
            "nearest_other_domain_mean_distance_b": b,
            "silhouette": score,
        })
    metrics = {
        "mean_within_domain_cosine_similarity": math.fsum(within) / len(within),
        "mean_cross_domain_cosine_similarity": math.fsum(cross) / len(cross),
        "cosine_distance_silhouette_score": math.fsum(x["silhouette"] for x in sample_scores) / n,
    }
    return {
        "schema_version": 1,
        "counts": {
            "datasets": n,
            "domains": len(groups),
            "within_domain_pairs": len(within),
            "cross_domain_pairs": len(cross),
            "singleton_domains": sum(len(members) == 1 for members in groups.values()),
        },
        "domain_sizes": dict(sorted(Counter(domains).items())),
        "metrics": metrics,
        "metrics_3dp": {name: f"{value:.3f}" for name, value in metrics.items()},
        "conventions": {
            "pair_aggregation": "Unweighted mean over unique unordered pairs i < j; diagonal excluded",
            "distance": "D_ij = 1 - S_ij",
            "silhouette_aggregation": "Unweighted mean over every dataset",
            "singleton_silhouette": 0.0,
            "zero_denominator_silhouette": 0.0,
            "domain_grouping": "Predefined labels supplied in dataset_domains.csv; no clustering is fitted",
        },
        "per_dataset": sample_scores,
    }


def _input_record(path):
    path = Path(path)
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--domains", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()
    try:
        labels, matrix = load_matrix(args.matrix)
        domains = load_domains(args.domains, labels)
        result = evaluate(labels, matrix, domains)
        result["inputs"] = {"matrix": _input_record(args.matrix), "domains": _input_record(args.domains)}
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result["metrics_3dp"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
