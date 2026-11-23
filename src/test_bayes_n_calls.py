# -*- coding: utf-8 -*-
# Copyright (c) 2026 Southeast University.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from hardware import HW
from new_matmul_threemode import Matmul
from utils import Tensor, data_type_dict


def run_budget(n_calls):
    model = Matmul(data_type=data_type_dict["fp16"])
    model(Tensor([1096, 1600]), Tensor([1600, 1096]))
    calls = [0]
    simulate = model.simulate

    def count_simulations(*args, **kwargs):
        calls[0] += 1
        return simulate(*args, **kwargs)

    model.simulate = count_simulations
    model.compile_and_simulate(HW, compile_mode="bayes", n_calls=n_calls)
    return calls[0]


def main():
    for n_calls in (10, 15):
        actual_calls = run_budget(n_calls)
        assert actual_calls == n_calls, (n_calls, actual_calls)


if __name__ == "__main__":
    main()
