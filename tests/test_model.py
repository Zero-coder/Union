"""Unit tests for model operations, gradients, and output shapes."""

import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from models.Union import Model
from union.model import (
    FrequencyAxisInteraction, SeriesTokenizer, TCILinearExpert, TCIMoE,
    UnionConfig, UnionModel,
)


class UnionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        torch.set_num_threads(1)
        self.config = UnionConfig(
            seq_len=8, pred_len=5, num_vars=2, num_classes=5,
            patch_len=4, stride=2, d_model=8, depth=1, n_heads=2,
            num_experts=3, top_k=2, prompt_num=2, dropout=0.0,
        )

    def test_variable_lengths_channels_and_task_output_shapes(self):
        model = UnionModel(self.config).eval()
        for length, variables in ((1, 1), (23, 5)):
            with self.subTest(length=length, variables=variables):
                x = torch.randn(2, length, variables)
                outputs = model(x, task="forecast", pred_len=13)
                self.assertEqual(outputs["forecast"].shape, (2, 13, variables))
                self.assertTrue(torch.isfinite(outputs["forecast"]).all())
                reconstructed = model(x, task="impute")["reconstruction"]
                self.assertEqual(reconstructed.shape, x.shape)
                anomaly = model(x, task="anomaly")
                self.assertEqual(anomaly["score"].shape, (2, length))
                torch.testing.assert_close(anomaly["score"], (anomaly["reconstruction"] - x).square().mean(-1))
                self.assertEqual(model(x, task="classify", num_classes=3)["logits"].shape, (2, 3))

    def test_tokenizer_keeps_trailing_observations_and_all_variables(self):
        tokenizer = SeriesTokenizer(self.config)
        x = torch.randn(2, 9, 7, requires_grad=True)
        tokens = tokenizer(x)
        self.assertEqual(tokens.shape, (2, 7, 8, 8))  # four data patches + two prompts + CLS/GEN
        tokens[:, :, -2, 0].sum().backward()
        self.assertGreater(x.grad[:, -1, :].abs().sum().item(), 0.0)
        self.assertTrue((x.grad[:, -1, :].abs().sum(0) > 0).all())

    def test_only_selected_dataset_prompt_receives_gradients(self):
        model = UnionModel(self.config)
        seen = model.register_dataset("Dataset.A")
        other = model.register_dataset("Dataset.B")
        self.assertIs(seen, model.register_dataset("Dataset.A"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        self.assertTrue(any(seen is p for group in optimizer.param_groups for p in group["params"]))
        loss = model(torch.randn(2, 11, 3), dataset_id="Dataset.A")["forecast"].square().mean()
        loss.backward()
        self.assertIsNotNone(seen.grad)
        self.assertGreater(seen.grad.abs().sum().item(), 0.0)
        self.assertIsNone(other.grad)
        self.assertIsNone(model.tokenizer.get_prompt("default").grad)
        for stage in (model.blocks[0].freq.channel, model.blocks[0].freq.temporal):
            gradient = stage.encoder.attn.in_proj_weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_unknown_dataset_requires_explicit_registration(self):
        model = UnionModel(self.config)
        with self.assertRaisesRegex(ValueError, "register_dataset"):
            model(torch.randn(1, 8, 2), dataset_id="unregistered")

    def test_fixed_kernel_has_normalized_distance_and_learnable_base_weights(self):
        expert = TCILinearExpert(8, 6, 0.0, kernel_tau=0.2)
        self.assertNotIn("kernel_tau", dict(expert.named_parameters()))
        self.assertFalse(expert.kernel_tau.requires_grad)
        for length in (1, 5, 13):
            kernel = expert.temporal_kernel(length)
            torch.testing.assert_close(kernel.diagonal(), torch.ones(length))
            torch.testing.assert_close(kernel, kernel.T)
            if length > 1:
                self.assertAlmostEqual(kernel[0, -1].item(), math.exp(-5), places=6)
        expert(torch.randn(2, 3, 9, 8)).square().mean().backward()
        self.assertGreater(expert.weight.grad.abs().sum().item(), 0.0)

    def test_moe_executes_only_routed_experts(self):
        moe = TCIMoE(8, 6, 4, 2, 0.0)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.bias.copy_(torch.tensor([10.0, 5.0, -5.0, -10.0]))
        calls = [0, 0, 0, 0]
        handles = []
        for index, expert in enumerate(moe.experts):
            def record_call(module, inputs, output, expert_id=index):
                calls[expert_id] += inputs[0].size(0)
            handles.append(expert.register_forward_hook(record_call))
        result = moe(torch.randn(3, 2, 7, 8))
        result.square().mean().backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [3, 3, 0, 0])
        self.assertIsNone(moe.experts[2].weight.grad)
        self.assertIsNone(moe.experts[3].weight.grad)

    def test_frequency_sampling_preserves_axis_and_frequency_locations(self):
        stage = FrequencyAxisInteraction(8, 2, 0.0, max_modes=2).eval()
        stage.encoder = nn.Identity()
        x = torch.randn(2, 7, 11, 8)
        for axis in (1, 2):
            with self.subTest(axis=axis):
                torch.manual_seed(43)
                output = stage(x, axis=axis)
                self.assertEqual(output.shape, x.shape)
                before = torch.fft.rfft(x, dim=axis)
                after = torch.fft.rfft(output, dim=axis)
                other_axes = tuple(i for i in range(4) if i != axis)
                selected = after.abs().sum(dim=other_axes) > 1e-4
                self.assertEqual(selected.sum().item(), 2)
                torch.testing.assert_close(after.movedim(axis, 0)[selected], before.movedim(axis, 0)[selected])
                self.assertLess(after.movedim(axis, 0)[~selected].abs().max().item(), 1e-5)

    def test_frequency_attention_is_not_pointwise_time_scaling(self):
        stage = FrequencyAxisInteraction(8, 2, 0.0, max_modes=64).eval()
        x = torch.randn(2, 3, 11, 8)
        perturbed = x.clone()
        perturbed[:, :, 2, 0] += 1.0
        delta = stage(perturbed, axis=2) - stage(x, axis=2)
        delta[:, :, 2] = 0
        self.assertGreater(delta.abs().max().item(), 1e-4)

    def test_frequency_eval_sampling_is_reproducible_when_seed_is_reset(self):
        stage = FrequencyAxisInteraction(8, 2, 0.0, max_modes=2).eval()
        x = torch.randn(2, 3, 31, 8)
        torch.manual_seed(7)
        a = stage(x, axis=2)
        torch.manual_seed(7)
        b = stage(x, axis=2)
        torch.testing.assert_close(a, b)

    def test_masked_values_do_not_leak_into_reconstruction(self):
        model = UnionModel(self.config).eval()
        x = torch.randn(2, 9, 3)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 3:6] = False
        modified = x.clone()
        modified[~mask] = float("nan")
        a = model(x, task="impute", mask=mask)["reconstruction"]
        b = model(modified, task="impute", mask=mask)["reconstruction"]
        torch.testing.assert_close(a, b)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model(modified, task="impute")

    def test_wrapper_registers_shared_dataset_prompt_and_task_dimensions(self):
        args = SimpleNamespace(patch_len=4, stride=2, d_model=8, e_layers=1,
                               n_heads=2, dropout=0.0, prompt_num=2)
        tasks = [
            ["short", {"dataset": "ECL", "seq_len": 8, "pred_len": 3, "enc_in": 2,
                       "task_name": "long_term_forecast"}],
            ["long", {"dataset": "ECL", "seq_len": 17, "pred_len": 9, "enc_in": 2,
                      "task_name": "long_term_forecast"}],
            ["cls", {"dataset": "ECG", "seq_len": 5, "enc_in": 1, "num_class": 3,
                     "task_name": "classification"}],
        ]
        wrapper = Model(args, tasks).eval()
        self.assertEqual(wrapper.dataset_ids, ["ECL", "ECL", "ECG"])
        self.assertEqual(len(wrapper.model.tokenizer.prompt_tokens), 3)  # default, ECL, ECG
        self.assertEqual(wrapper(torch.randn(2, 8, 2), task_id=0)["forecast"].shape, (2, 3, 2))
        self.assertEqual(wrapper(torch.randn(2, 17, 2), task_id=1)["forecast"].shape, (2, 9, 2))
        self.assertEqual(wrapper(torch.randn(2, 5, 1), task_id=2)["logits"].shape, (2, 3))
        pretrain = Model(args, tasks, pretrain=True).eval()
        self.assertEqual(pretrain(torch.randn(2, 5, 1), task_id=2)["reconstruction"].shape, (2, 5, 1))

    def test_invalid_configuration_and_task_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "stride"):
            UnionConfig(seq_len=8, pred_len=4, num_vars=1, patch_len=2, stride=3)
        with self.assertRaisesRegex(ValueError, "top_k"):
            UnionConfig(seq_len=8, pred_len=4, num_vars=1, top_k=5)
        model = UnionModel(self.config)
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            model(torch.randn(1, 8, 2), task="misspelled")

    def test_checkpoint_reference_length_allows_new_runtime_length(self):
        args = SimpleNamespace(patch_len=4, stride=2, d_model=8, e_layers=1,
                               n_heads=2, dropout=0.0, prompt_num=2,
                               reference_seq_len=8)
        source_tasks = [["forecast", {"dataset": "series", "seq_len": 8,
                         "pred_len": 3, "enc_in": 2, "task_name": "long_term_forecast"}]]
        target_tasks = [["forecast", {"dataset": "series", "seq_len": 19,
                         "pred_len": 11, "enc_in": 5, "task_name": "long_term_forecast"}]]
        source = Model(args, source_tasks).eval()
        target = Model(args, target_tasks).eval()
        target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(target.model.config.seq_len, 8)
        self.assertEqual(target(torch.randn(2, 19, 5))["forecast"].shape, (2, 11, 5))


if __name__ == "__main__":
    unittest.main()
