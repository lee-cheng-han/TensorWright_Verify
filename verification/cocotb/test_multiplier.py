from __future__ import annotations

import random

import cocotb
from cocotb.triggers import Timer


@cocotb.test()  # type: ignore[untyped-decorator]
async def multiplier_matches_signed_int8_product(dut) -> None:  # type: ignore[no-untyped-def]
    cases = [(-128, -128), (-128, 127), (127, 127), (0, -128), (1, -1)]
    random_source = random.Random(0x5A17)
    cases.extend(
        (random_source.randint(-128, 127), random_source.randint(-128, 127))
        for _ in range(500)
    )
    for activation, weight in cases:
        dut.activation_i.value = activation
        dut.weight_i.value = weight
        await Timer(1, unit="ns")
        actual = int(dut.product_o.value)
        if actual >= 1 << 15:
            actual -= 1 << 16
        assert actual == activation * weight
