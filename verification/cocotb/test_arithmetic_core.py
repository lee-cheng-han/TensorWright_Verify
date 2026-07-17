from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from tensorwright.reference import requantize_int32

LANES = 9


def _pack(values: list[int], width: int = 8) -> int:
    packed = 0
    mask = (1 << width) - 1
    for index, value in enumerate(values):
        packed |= (value & mask) << (index * width)
    return packed


async def _reset(dut) -> None:  # type: ignore[no-untyped-def]
    dut.rst_ni.value = 0
    dut.valid_i.value = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1


@cocotb.test()  # type: ignore[untyped-decorator]
async def arithmetic_core_matches_accumulated_dot_products(dut) -> None:  # type: ignore[no-untyped-def]
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await _reset(dut)
    random_source = random.Random(0xC0DE5)

    for _ in range(150):
        cycle_count = random_source.randint(1, 8)
        accumulator = 0
        bias = random_source.randint(-10_000, 10_000)
        multiplier = random_source.randint(0, (1 << 31) - 1)
        shift = random_source.randint(0, 45)
        relu = bool(random_source.getrandbits(1))
        for cycle in range(cycle_count):
            activations = [random_source.randint(-128, 127) for _ in range(LANES)]
            weights = [random_source.randint(-128, 127) for _ in range(LANES)]
            accumulator += sum(
                activation * weight
                for activation, weight in zip(activations, weights, strict=True)
            )
            dut.activations_i.value = _pack(activations)
            dut.weights_i.value = _pack(weights)
            dut.clear_i.value = cycle == 0
            dut.last_i.value = cycle == cycle_count - 1
            dut.bias_i.value = bias
            dut.multiplier_i.value = multiplier
            dut.shift_i.value = shift
            dut.relu_i.value = relu
            dut.valid_i.value = 1
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
    dut.valid_i.value = 0
