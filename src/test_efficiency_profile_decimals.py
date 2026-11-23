# Copyright (c) 2026 Southeast University.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression coverage for malformed decimal values in bandwidth profiles."""

import csv
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from new_matmul_threemode import Matmul  # noqa: E402
from utils import Tensor, data_type_dict  # noqa: E402


class EfficiencyProfileDecimalTests(unittest.TestCase):
    @staticmethod
    def _simulate_profile_point(profile, legacy_corruption=False):
        shape, mapping, target = profile
        model = Matmul(data_type_dict["fp16"])
        model(Tensor([shape[0], shape[2]]), Tensor([shape[2], shape[1]]))
        if legacy_corruption:
            if target == "A":
                model.L12L0A_eff_list = [
                    (size, 997183.0 if size == 48.5 else efficiency)
                    for size, efficiency in model.L12L0A_eff_list
                ]
            else:
                model.L12L0B_eff_list = [
                    (size, 997964.0 if size == 28.5 else efficiency)
                    for size, efficiency in model.L12L0B_eff_list
                ]
        return model.simulate(model.computational_graph, Matmul.Mapping(*mapping))

    def test_repaired_points_have_decimal_bandwidth_and_efficiency(self):
        expected = {
            "l12L0A_efficiency.csv": (48.5, 438.760520, 0.997183),
            "l12L0B_efficiency.csv": (28.5, 219.55208, 0.997964),
        }
        for filename, (traffic, bandwidth, efficiency) in expected.items():
            with self.subTest(filename=filename):
                with (ROOT / "data" / filename).open(encoding="utf-8", newline="") as handle:
                    rows = {float(row["traffic_size(KB)"]): row for row in csv.DictReader(handle)}
                row = rows[traffic]
                self.assertAlmostEqual(float(row["bandwidth(GB/s)"]), bandwidth)
                actual_efficiency = float(row["efficiency"])
                self.assertAlmostEqual(actual_efficiency, efficiency)
                self.assertGreaterEqual(actual_efficiency, 0.0)
                self.assertLessEqual(actual_efficiency, 1.0)

    def test_repaired_points_change_their_tile_read_estimate(self):
        profiles = [
            ((48, 16, 528), (48, 16, 528, "mnk"), "A", 337.7220550290167, 266.86676728745203),
            ((16, 48, 304), (16, 48, 304, "mnk"), "B", 319.91158507812816, 116.02985752801251),
        ]
        for shape, mapping, target, expected, legacy in profiles:
            profile = (shape, mapping, target)
            with self.subTest(profile=target):
                repaired_cycles = self._simulate_profile_point(profile)
                legacy_cycles = self._simulate_profile_point(profile, legacy_corruption=True)
                self.assertTrue(math.isfinite(repaired_cycles))
                self.assertAlmostEqual(repaired_cycles, expected, places=9)
                self.assertAlmostEqual(legacy_cycles, legacy, places=9)
                self.assertGreater(repaired_cycles, legacy_cycles)

    def test_exhaustive_search_control_is_stable(self):
        def run():
            model = Matmul(data_type_dict["fp16"])
            model(Tensor([256, 256]), Tensor([256, 256]))
            model.compile_and_simulate(compile_mode="exhaustive")
            return model.best_cycle_count, model.best_mapping

        cycles, mapping = run()
        control_cycles, control_mapping = run()
        self.assertTrue(math.isfinite(cycles))
        self.assertGreater(cycles, 0.0)
        self.assertAlmostEqual(cycles, 3995.2052957023716, places=9)
        self.assertEqual(cycles, control_cycles)
        self.assertEqual(
            (mapping.l1_tile_M, mapping.l1_tile_N, mapping.l1_tile_K, mapping.l1_loop_order),
            (
                control_mapping.l1_tile_M,
                control_mapping.l1_tile_N,
                control_mapping.l1_tile_K,
                control_mapping.l1_loop_order,
            ),
        )


if __name__ == "__main__":
    unittest.main()
