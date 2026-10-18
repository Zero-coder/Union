"""Analytical fixtures and input-validation tests for the evaluator."""

import math
import tempfile
import unittest
from pathlib import Path

from evaluate_prompt_structure import evaluate, load_domains, load_matrix


class EvaluationTests(unittest.TestCase):
    def test_separated_groups_and_singleton(self):
        # Unit vectors: A=(1,0) twice, B=(0,1) twice, C=(-1,0).
        # Within-group distances are zero; four nonsingletons score 1.
        matrix = [
            [1, 1, 0, 0, -1],
            [1, 1, 0, 0, -1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [-1, -1, 0, 0, 1],
        ]
        result = evaluate(list("abcde"), matrix, ["A", "A", "B", "B", "C"])
        self.assertEqual(result["counts"], {
            "datasets": 5, "domains": 3, "within_domain_pairs": 2,
            "cross_domain_pairs": 8, "singleton_domains": 1,
        })
        self.assertEqual(result["metrics"]["mean_within_domain_cosine_similarity"], 1)
        self.assertEqual(result["metrics"]["mean_cross_domain_cosine_similarity"], -0.25)
        self.assertEqual(result["metrics"]["cosine_distance_silhouette_score"], 0.8)
        self.assertEqual(result["per_dataset"][-1]["silhouette"], 0)

    def test_overlapping_groups_have_negative_silhouette(self):
        # Both groups contain +x and -x. a=2, b=1 for each point.
        matrix = [[1, -1, 1, -1], [-1, 1, -1, 1],
                  [1, -1, 1, -1], [-1, 1, -1, 1]]
        result = evaluate(list("abcd"), matrix, ["A", "A", "B", "B"])
        self.assertEqual(result["metrics"]["cosine_distance_silhouette_score"], -0.5)

    def test_identical_points_zero_denominator(self):
        result = evaluate(list("abc"), [[1] * 3 for _ in range(3)], ["A", "A", "B"])
        self.assertEqual(result["metrics"]["cosine_distance_silhouette_score"], 0)

    def test_invalid_matrices_rejected(self):
        cases = [
            ([[1, 0.1, 0], [0.2, 1, 0], [0, 0, 1]], "symmetric"),
            ([[1, math.nan, 0], [math.nan, 1, 0], [0, 0, 1]], "finite"),
            ([[1, 1.1, 0], [1.1, 1, 0], [0, 0, 1]], "finite"),
            ([[0.9, 0, 0], [0, 1, 0], [0, 0, 1]], "Diagonal"),
        ]
        for matrix, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                evaluate(list("abc"), matrix, ["A", "A", "B"])

    def test_invalid_groupings_rejected(self):
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for domains in (["A", "A", "A"], ["A", "B", "C"], ["A", "", "B"]):
            with self.subTest(domains=domains), self.assertRaises(ValueError):
                evaluate(list("abc"), matrix, domains)

    def test_csv_name_alignment_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "domains.csv"
            path.write_text("dataset,domain\nc,B\na,A\nb,A\n", encoding="utf-8")
            self.assertEqual(load_domains(path, list("abc")), ["A", "A", "B"])
            for text in ("dataset,domain\na,A\na,B\n", "dataset,domain\na,A\nx,B\n"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_domains(path, list("abc"))
            path.write_text("dataset,a,a\na,1,0\na,0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_matrix(path)
            path.write_text("dataset,a,b\nb,1,0\na,0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "order"):
                load_matrix(path)


if __name__ == "__main__":
    unittest.main()
