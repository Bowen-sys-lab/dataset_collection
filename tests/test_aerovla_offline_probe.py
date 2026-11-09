import unittest
import importlib.util


class AeroVLAActionParsingTests(unittest.TestCase):
    def parse(self, value):
        from scripts.aerovla_offline_probe import parse_action
        return parse_action(value)

    def test_action_bins_have_documented_physical_units(self):
        action = self.parse('98 49 98')
        self.assertEqual(action['forward_m'], 5.0)
        self.assertEqual(action['down_m'], 0.0)
        self.assertAlmostEqual(action['yaw_rad'], 1.1)
        self.assertFalse(action['official_termination'])

    def test_land_token_is_recorded_independently_of_motion(self):
        action = self.parse('49 49 49 LAND')
        self.assertEqual(action['termination_reason'], 'LAND_TOKEN')
        self.assertGreater(action['forward_m'], 0)

    def test_official_near_zero_rule_is_preserved_as_diagnostic(self):
        action = self.parse('00 49 49')
        self.assertEqual(action['termination_reason'], 'NEAR_ZERO_ACTION')

    def test_malformed_output_cannot_become_successful_zero_motion(self):
        for value in ['', 'hello', '00 49', '99 49 49', '-1 49 49', '0.5 49 49',
                      'Fly 3 meters then Action: 00 49 49', '00 49 49 12']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)


@unittest.skipUnless(importlib.util.find_spec('bitsandbytes') and importlib.util.find_spec('transformers'),
                     'Requires the independent model environment')
class AeroVLAQuantizationTests(unittest.TestCase):
    def test_saved_projector_remains_float_when_language_layers_are_quantized(self):
        import torch
        from torch import nn
        import bitsandbytes as bnb
        from transformers.integrations.bitsandbytes import replace_with_bnb_linear
        from scripts.aerovla_offline_probe import quantization_config

        model = nn.Module()
        model.projector = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model.language_model = nn.Sequential(nn.Linear(4, 4))
        model.lm_head = nn.Linear(4, 4)
        config = quantization_config()
        converted = replace_with_bnb_linear(model, quantization_config=config,
                                             modules_to_not_convert=config.llm_int8_skip_modules)
        self.assertIsInstance(converted.projector[0], nn.Linear)
        self.assertNotIsInstance(converted.projector[0], bnb.nn.Linear4bit)
        self.assertNotIsInstance(converted.lm_head, bnb.nn.Linear4bit)
        self.assertIsInstance(converted.language_model[0], bnb.nn.Linear4bit)
        from peft.utils.other import ModulesToSaveWrapper
        wrapper = ModulesToSaveWrapper(converted.projector, 'default')
        self.assertTrue(all(parameter.is_floating_point() for parameter in wrapper.parameters()))


if __name__ == '__main__':
    unittest.main()
