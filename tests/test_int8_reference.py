from __future__ import annotations

import random
import unittest

from tensorwright.reference import (
    Conv2DConfig,
    conv2d_int8,
    multiply_int8,
    quantize_symmetric,
    requantize_int32,
    round_shift_right,
    saturate_int8,
)


class ArithmeticTest(unittest.TestCase):
    def test_int8_product_edges(self) -> None:
        self.assertEqual(multiply_int8(127, 127), 16129)
        self.assertEqual(multiply_int8(-128, -128), 16384)
        self.assertEqual(multiply_int8(127, -128), -16256)

    def test_zero_inputs_and_weights(self) -> None:
        self.assertEqual(multiply_int8(0, 127), 0)
        self.assertEqual(multiply_int8(-128, 0), 0)

    def test_saturation(self) -> None:
        self.assertEqual(saturate_int8(1000), 127)
        self.assertEqual(saturate_int8(-1000), -128)

    def test_symmetric_float_quantization(self) -> None:
        self.assertEqual(quantize_symmetric(0.5, 1.0), 1)
        self.assertEqual(quantize_symmetric(-0.5, 1.0), -1)
        self.assertEqual(quantize_symmetric(1000.0, 1.0), 127)
        self.assertEqual(quantize_symmetric(-1000.0, 1.0), -128)
        self.assertEqual(quantize_symmetric(1.0, 1e-320), 127)

    def test_shift_zero_large_and_halfway(self) -> None:
        self.assertEqual(round_shift_right(-17, 0), -17)
        self.assertEqual(round_shift_right(127, 20), 0)
        self.assertEqual(round_shift_right(3, 1), 2)
        self.assertEqual(round_shift_right(-3, 1), -2)
        self.assertEqual(round_shift_right(1, 1), 1)
        self.assertEqual(round_shift_right(-1, 1), -1)

    def test_requantization_order_and_relu(self) -> None:
        self.assertEqual(requantize_int32(10, 2, 3, 1), 18)
        self.assertEqual(requantize_int32(-10, 0, 1, 0, relu=True), 0)
        self.assertEqual(requantize_int32(1000, 0, 1, 0), 127)
        self.assertEqual(requantize_int32(-1000, 0, 1, 0), -128)

    def test_int32_accumulator_boundaries(self) -> None:
        self.assertEqual(requantize_int32((1 << 31) - 1, 0, 1, 31), 1)
        self.assertEqual(requantize_int32(-(1 << 31), 0, 1, 31), -1)
        with self.assertRaises(OverflowError):
            requantize_int32((1 << 31) - 1, 1, 1, 0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            multiply_int8(128, 1)
        with self.assertRaises(ValueError):
            quantize_symmetric(1.0, 0.0)
        with self.assertRaises(ValueError):
            round_shift_right(1, -1)
        with self.assertRaises(TypeError):
            requantize_int32(1, 0, True, 0)

    def test_seeded_random_requantization(self) -> None:
        seed = 0x7E115
        random_source = random.Random(seed)
        for _ in range(500):
            value = random_source.randint(-1_000_000, 1_000_000)
            multiplier = random_source.randint(0, 255)
            shift = random_source.randint(0, 16)
            product = value * multiplier
            if shift == 0:
                expected = product
            else:
                magnitude = (abs(product) + (1 << (shift - 1))) // (1 << shift)
                expected = magnitude if product >= 0 else -magnitude
            expected = min(127, max(-128, expected))
            case = {
                "seed": seed,
                "value": value,
                "multiplier": multiplier,
                "shift": shift,
            }
            with self.subTest(**case):
                self.assertEqual(
                    requantize_int32(value, 0, multiplier, shift), expected
                )


class ConvolutionTest(unittest.TestCase):
    def test_small_multichannel_convolution(self) -> None:
        inputs = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        ]
        weights = [
            [
                [[1, 0], [0, -1]],
                [[0, 1], [-1, 0]],
            ]
        ]
        self.assertEqual(
            conv2d_int8(inputs, weights, [0], [1], [0]),
            [[[-2, -2], [-2, -2]]],
        )

    def test_stride_padding_bias_and_relu(self) -> None:
        inputs = [[[-2, 1], [3, -4]]]
        weights = [[[[1]]]]
        output = conv2d_int8(
            inputs,
            weights,
            [1],
            [1],
            [0],
            Conv2DConfig(
                stride_height=2,
                stride_width=2,
                padding=(1, 1, 1, 1),
                relu=True,
            ),
        )
        self.assertEqual(output, [[[1, 1], [1, 0]]])

    def test_seeded_random_convolution_against_direct_sum(self) -> None:
        seed = 0xC011
        random_source = random.Random(seed)
        for _ in range(50):
            channels = random_source.randint(1, 3)
            height = random_source.randint(2, 5)
            width = random_source.randint(2, 5)
            kernel_height = random_source.randint(1, height)
            kernel_width = random_source.randint(1, width)
            inputs = [
                [
                    [random_source.randint(-5, 5) for _ in range(width)]
                    for _ in range(height)
                ]
                for _ in range(channels)
            ]
            weights = [
                [
                    [
                        [random_source.randint(-5, 5) for _ in range(kernel_width)]
                        for _ in range(kernel_height)
                    ]
                    for _ in range(channels)
                ]
            ]
            expected_rows: list[list[int]] = []
            for output_y in range(height - kernel_height + 1):
                row: list[int] = []
                for output_x in range(width - kernel_width + 1):
                    total = sum(
                        inputs[channel][output_y + kernel_y][output_x + kernel_x]
                        * weights[0][channel][kernel_y][kernel_x]
                        for channel in range(channels)
                        for kernel_y in range(kernel_height)
                        for kernel_x in range(kernel_width)
                    )
                    row.append(min(127, max(-128, total)))
                expected_rows.append(row)
            with self.subTest(seed=seed, shape=(channels, height, width)):
                self.assertEqual(
                    conv2d_int8(inputs, weights, [0], [1], [0]),
                    [expected_rows],
                )

    def test_shape_and_range_validation(self) -> None:
        with self.assertRaises(ValueError):
            conv2d_int8([[[1], [2, 3]]], [[[[1]]]], [0], [1], [0])
        with self.assertRaises(ValueError):
            conv2d_int8([[[128]]], [[[[1]]]], [0], [1], [0])
        with self.assertRaises(ValueError):
            conv2d_int8([[[1]]], [[[[1]]]], [], [1], [0])
