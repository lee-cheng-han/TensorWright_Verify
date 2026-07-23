from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from tensorwright.reference import requantize_int32


async def _reset(dut) -> None:  # type: ignore[no-untyped-def]
    dut.rst_ni.value = 0
    dut.valid_i.value = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1


@cocotb.test()  # type: ignore[untyped-decorator]
async def postprocess_matches_python_reference(dut) -> None:  # type: ignore[no-untyped-def]
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await _reset(dut)
    cases = [
        (0, 0, 1, 0, False),
        (3, 0, 1, 1, False),
        (-3, 0, 1, 1, False),
        (1000, 0, 1, 0, False),
        (-1000, 0, 1, 0, False),
        (-10, 0, 1, 0, True),
        ((1 << 31) - 1, 0, (1 << 30), 62, False),
        (-(1 << 31), 0, (1 << 30), 62, False),
    ]
    random_source = random.Random(0xA511)
    cases.extend(
        (
            random_source.randint(-1_000_000, 1_000_000),
            random_source.randint(-100_000, 100_000),
            random_source.randint(0, (1 << 31) - 1),
            random_source.randint(0, 70),
            bool(random_source.getrandbits(1)),
        )
        for _ in range(500)
    )

    for accumulator, bias, multiplier, shift, relu in cases:
        dut.accumulator_i.value = accumulator
        dut.bias_i.value = bias
        dut.multiplier_i.value = multiplier
        dut.shift_i.value = shift
        dut.relu_i.value = relu
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")
        dut.valid_i.value = 0
        while not int(dut.valid_o.value):
            await RisingEdge(dut.clk_i)
            await Timer(1, unit="ns")
        assert int(dut.valid_o.value) == 1
        assert int(dut.overflow_o.value) == 0
        actual = int(dut.result_o.value)
        if actual >= 128:
            actual -= 256
        expected = requantize_int32(accumulator, bias, multiplier, shift, relu=relu)
        assert actual == expected, (
            accumulator,
            bias,
            multiplier,
            shift,
            relu,
            expected,
            actual,
        )
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")
    dut.valid_i.value = 0
