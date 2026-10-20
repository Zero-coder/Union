"""Unit tests for dataset boundaries, training, and evaluation using temporary file fixtures."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from data_provider.data_factory import data_provider
from exp.exp_sup import Exp_All_Task, iter_task_batches, make_missing_mask
from exp.exp_pretrain import Exp_All_Task as PretrainExp
from run import build_parser, seed_everything, validate_args
from run_pretrain import build_parser as pretrain_parser
from union.data import load_dataset
from utils.losses import UnifiedMaskRecLoss, masked_mse


class DataTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        with (self.root / "series.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "a", "b"])
            writer.writerows([str(i), i, 2 * i] for i in range(80))
        self.forecast = dict(task_name="long_term_forecast", dataset="series",
                             root_path=str(self.root), data_path="series.csv", seq_len=8,
                             pred_len=4, enc_in=2, split_borders=[50, 65], scale=False)
        self.imputation = dict(self.forecast, task_name="imputation", pred_len=0)
        x = np.arange(6 * 8 * 2, dtype=np.float32).reshape(6, 8, 2) / 100
        np.savez(self.root / "classification.npz", x_train=x, y_train=np.arange(6) % 2,
                 x_val=x[:4], y_val=np.arange(4) % 2, x_test=x[:5], y_test=np.arange(5) % 2)
        self.classification = dict(task_name="classification", dataset="classification",
                                   root_path=str(self.root), data_path="classification.npz",
                                   seq_len=8, pred_len=0, enc_in=2, num_class=2)
        raw = np.arange(80, dtype=np.float32).reshape(40, 2) / 100
        np.savez(self.root / "anomaly.npz", train=raw, val=raw[:20], test=raw[:24],
                 test_labels=(np.arange(24) >= 20).astype(np.int64))
        self.anomaly = dict(task_name="anomaly_detection", dataset="anomaly",
                            root_path=str(self.root), data_path="anomaly.npz",
                            seq_len=8, pred_len=0, enc_in=2, window_stride=5)

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, configs, pretrain=False):
        path = self.root / "tasks.yaml"
        path.write_text(yaml.safe_dump({"task_dataset": configs}), encoding="utf-8")
        parser = pretrain_parser() if pretrain else build_parser()
        args = parser.parse_args(["--is_training", "1", "--model_id", "test",
                                  "--task_data_config_path", str(path), "--checkpoints", str(self.root / "checkpoints"),
                                  "--device", "cpu", "--train_epochs", "1", "--batch_size", "16",
                                  "--d_model", "8", "--n_heads", "2", "--e_layers", "1",
                                  "--patch_len", "4", "--stride", "4", "--prompt_num", "2",
                                  "--num_experts", "2", "--top_k", "1", "--max_freq_modes", "4",
                                  "--dropout", "0", "--fix_seed", "12", "--acc_it", "2"])
        validate_args(args)
        return args

    def test_forecast_targets_respect_split_boundaries(self):
        train = load_dataset(self.forecast, "train")
        val = load_dataset(self.forecast, "val")
        test = load_dataset(self.forecast, "test")
        self.assertEqual(train[-1]["y"][-1, 0].item(), 49)
        self.assertEqual(val[0]["x"][0, 0].item(), 42)
        self.assertEqual(val[0]["y"][0, 0].item(), 50)
        self.assertEqual(val[-1]["y"][-1, 0].item(), 64)
        self.assertEqual(test[0]["y"][0, 0].item(), 65)
        self.assertEqual(test[-1]["y"][-1, 0].item(), 79)

    def test_scaler_uses_only_training_rows(self):
        config = dict(self.forecast, scale=True)
        test = load_dataset(config, "test")
        expected = (57 - np.arange(50).mean()) / np.arange(50).std()
        self.assertAlmostEqual(test[0]["x"][0, 0].item(), expected, places=6)
        train = load_dataset(config, "train")
        self.assertAlmostEqual(float(train.values[:, 0].mean()), 0.0, places=6)

    def test_fraction_and_shuffle_are_seeded(self):
        left = load_dataset(self.forecast, "train", 0.2, seed=41)
        right = load_dataset(self.forecast, "train", 0.2, seed=41)
        different = load_dataset(self.forecast, "train", 0.2, seed=42)
        self.assertEqual(left.indices, right.indices)
        self.assertNotEqual(left.indices, different.indices)
        self.assertEqual(len(load_dataset(self.forecast, "test", 0.2)), len(load_dataset(self.forecast, "test")))
        args = self.args({"forecast": self.forecast})
        _, a = data_provider(args, self.forecast, "train")
        _, b = data_provider(args, self.forecast, "train")
        self.assertTrue(torch.equal(next(iter(a))["x"], next(iter(b))["x"]))

    def test_fewshot_scaler_uses_only_selected_samples(self):
        selection = np.random.default_rng(41).permutation(6)[:1]
        with np.load(self.root / "classification.npz") as arrays:
            fitting = arrays["x_train"][selection]
            expected = (arrays["x_test"] - fitting.mean(axis=(0, 1))) / fitting.std(axis=(0, 1))
        actual = load_dataset(self.classification, "test", 0.2, seed=41)
        np.testing.assert_allclose(actual.x, expected, rtol=1e-6, atol=1e-6)

    def test_npz_classification_contract(self):
        test = load_dataset(self.classification, "test")
        self.assertEqual(len(test), 5)
        self.assertEqual(tuple(test[0]["x"].shape), (8, 2))
        self.assertEqual(test[0]["y"].dtype, torch.long)
        invalid = dict(self.classification, enc_in=3)
        with self.assertRaisesRegex(ValueError, "channels"):
            load_dataset(invalid, "train")

    def test_file_errors_never_substitute_data(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(dict(self.forecast, data_path="missing.csv"), "train")
        missing_path = dict(self.classification)
        missing_path.pop("data_path")
        with self.assertRaisesRegex(ValueError, "conversion"):
            load_dataset(missing_path, "train")

    def test_masked_loss_excludes_visible_values(self):
        pred = torch.tensor([[[3.0], [1000.0]]], requires_grad=True)
        target = torch.tensor([[[1.0], [0.0]]])
        mask = torch.tensor([[[True], [False]]])
        loss = masked_mse(pred, target, mask)
        self.assertEqual(loss.item(), 4.0)
        loss.backward()
        self.assertEqual(pred.grad[0, 1, 0].item(), 0.0)
        self.assertEqual(UnifiedMaskRecLoss()({"reconstruction": pred}, target, mask)["loss"].item(), 4.0)
        generator = torch.Generator().manual_seed(10)
        sampled = make_missing_mask(torch.ones(5, 1, 1), 0.001, generator)
        self.assertTrue(sampled.reshape(5, -1).any(dim=1).all())

    def test_round_robin_keeps_all_batches(self):
        self.assertEqual(list(iter_task_batches([[1, 2, 3], [4]])), [(1, 0), (4, 1), (2, 0), (3, 0)])

    def test_train_save_load_and_full_test_for_all_tasks(self):
        args = self.args({"forecast": self.forecast, "imputation": self.imputation,
                          "classification": self.classification, "anomaly": self.anomaly})
        args.batch_size = 3
        args.max_freq_modes = 1
        args.subsample_pct = 0.2
        seed_everything(args.fix_seed)
        exp = Exp_All_Task(args)
        exp.train("integration")
        rng_before = torch.random.get_rng_state().clone()
        result = exp.test("integration")
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertEqual(set(result), {"forecast", "imputation", "classification", "anomaly"})
        self.assertEqual(result["classification"]["samples"], 5)
        self.assertEqual(result["forecast"]["samples"], 12)
        self.assertEqual(result["anomaly"]["test_points"], 24)
        self.assertFalse(result["anomaly"]["point_adjustment"])
        self.assertTrue(np.isfinite(result["imputation"]["mse"]))
        best = {key: value.clone() for key, value in exp.model.state_dict().items()}
        with torch.no_grad():
            next(exp.model.parameters()).add_(10)
        restored = exp.test("integration", load_pretrain=True)
        self.assertEqual(result, restored)
        self.assertTrue(all(torch.equal(best[key], value) for key, value in exp.model.state_dict().items()))
        persisted = json.loads((self.root / "checkpoints/integration/test_metrics.json").read_text())
        self.assertEqual(persisted["seed"], 12)

    def test_pretraining_runs_masked_reconstruction(self):
        args = self.args({"forecast": self.forecast}, pretrain=True)
        args.subsample_pct = 0.1
        seed_everything(args.fix_seed)
        exp = PretrainExp(args)
        batch = next(iter(exp._get_data("train")[0][1]))
        captured = []
        hook = exp.model.register_forward_pre_hook(lambda module, inputs, kwargs: captured.append(kwargs["x_enc"].detach().clone()), with_kwargs=True)
        loss, pred, target, missing = exp._batch(batch, 0, training=True)
        hook.remove()
        self.assertEqual(tuple(pred.shape), tuple(batch["x"].shape))
        self.assertTrue(torch.equal(target, batch["x"]))
        self.assertTrue((captured[0][missing] == 0).all())
        self.assertGreater(int(missing.sum()), 0)
        self.assertTrue(torch.isfinite(loss))
        exp.train("pretrain")
        self.assertTrue((self.root / "checkpoints/pretrain/checkpoint.pth").is_file())

    def test_unsupported_cli_modes_fail_explicitly(self):
        args = self.args({"forecast": self.forecast})
        args.prompt_tune_epoch = 2
        with self.assertRaisesRegex(ValueError, "prompt_tune_epoch"):
            validate_args(args)

    def test_checkpoint_rejects_same_shape_architecture_mismatch(self):
        from dataclasses import asdict
        args = self.args({"forecast": self.forecast})
        exp = Exp_All_Task(args)
        config = asdict(exp.model.model.config)
        config["n_heads"] = 4
        path = self.root / "different_heads.pth"
        torch.save({"model": exp.model.state_dict(), "model_config": config}, path)
        with self.assertRaisesRegex(ValueError, "n_heads"):
            exp._load_checkpoint(path)

    def test_transfer_reuses_backbone_and_initializes_new_dataset_prompt(self):
        from dataclasses import asdict
        source = Exp_All_Task(self.args({"source": self.forecast}))
        with torch.no_grad():
            source.model.model.norm.weight.fill_(2.0)
        path = self.root / "source.pth"
        torch.save({"model": source.model.state_dict(), "model_config": asdict(source.model.model.config)}, path)
        args = self.args({"target": dict(self.forecast, dataset="new_series")})
        seed_everything(23)
        fresh = Exp_All_Task(args)
        initial_prompt = fresh.model.model.tokenizer.get_prompt("new_series").detach().clone()
        args.pretrained_weight = str(path)
        seed_everything(23)
        target = Exp_All_Task(args)
        self.assertTrue(torch.equal(target.model.model.norm.weight, source.model.model.norm.weight))
        self.assertTrue(torch.equal(target.model.model.tokenizer.get_prompt("new_series"), initial_prompt))


if __name__ == "__main__":
    unittest.main()
