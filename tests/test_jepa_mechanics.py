import unittest

from tinygrad import Device

from experiments.jepa_mechanics import _patches, compare


class JEPAMechanicsTest(unittest.TestCase):
    def test_position_conditioned_prediction_uses_an_ema_target(self) -> None:
        observation = compare(
            Device.DEFAULT,
            seed=0,
            steps=80,
            samples=16,
            hidden_features=8,
            learning_rate=0.01,
            ema_decay=0.99,
        )

        self.assertLess(observation.aligned_loss, observation.initial_loss / 5)
        self.assertGreater(observation.shuffled_target_loss, observation.aligned_loss * 1.5)
        self.assertGreater(observation.unconditioned_loss, observation.aligned_loss * 1.4)
        self.assertGreater(observation.target_sample_std, 0.1)
        self.assertGreater(observation.target_parameter_delta, 0)
        self.assertEqual(observation.target_gradient, 0)

    def test_fixture_requires_a_square_sample_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            _patches(6, Device.DEFAULT)


if __name__ == "__main__":
    unittest.main()
