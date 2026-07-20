import unittest

import torch
from torch import nn

from detection.qa_probe import QAProbe
from scripts.train_qa_fixed_mlp_experiment import (
    ExperimentConfig,
    FixedMLP,
)


class FixedQAExperimentMLPTests(unittest.TestCase):
    def test_default_qa_probe_uses_the_same_bn_relu_structure(self):
        layers = list(QAProbe(17, (128, 64, 32), 0.3).network)
        self.assertEqual(
            [type(layer) for layer in layers],
            [
                nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout,
                nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout,
                nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout,
                nn.Linear,
            ],
        )

    def test_exact_requested_network_structure(self):
        config = ExperimentConfig()
        model = FixedMLP(17, config)
        layers = list(model.network)

        self.assertEqual(config.hidden_sizes, (128, 64, 32))
        self.assertEqual(config.dropout, 0.3)
        self.assertEqual(
            [type(layer) for layer in layers],
            [
                nn.Linear,
                nn.BatchNorm1d,
                nn.ReLU,
                nn.Dropout,
                nn.Linear,
                nn.BatchNorm1d,
                nn.ReLU,
                nn.Dropout,
                nn.Linear,
                nn.BatchNorm1d,
                nn.ReLU,
                nn.Dropout,
                nn.Linear,
            ],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in layers if isinstance(layer, nn.Linear)],
            [(17, 128), (128, 64), (64, 32), (32, 1)],
        )
        self.assertTrue(
            all(layer.p == 0.3 for layer in layers if isinstance(layer, nn.Dropout))
        )

    def test_initialization_and_fixed_training_protocol(self):
        config = ExperimentConfig()
        model = FixedMLP(5, config)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                self.assertTrue(torch.isfinite(module.weight).all())
            elif isinstance(module, nn.BatchNorm1d):
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

        self.assertEqual(config.optimizer, "Adam")
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 1e-5)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.max_epochs, 100)
        self.assertEqual(config.scheduler_patience, 5)
        self.assertEqual(config.early_stopping_patience, 10)
        self.assertEqual(config.checkpoint_selection, "minimum_train_loss")
        self.assertEqual(config.threshold, 0.5)
        self.assertEqual(config.threshold_selection, "fixed")


if __name__ == "__main__":
    unittest.main()
