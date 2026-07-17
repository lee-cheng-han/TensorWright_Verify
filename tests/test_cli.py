from __future__ import annotations

import contextlib
import io
import unittest

from tensorwright.cli import main


class CliTest(unittest.TestCase):
    def test_version(self) -> None:
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            self.assertRaisesRegex(SystemExit, "0"),
        ):
            main(["--version"])

        self.assertEqual(output.getvalue(), "tensorwright 0.0.0\n")

    def test_help(self) -> None:
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            self.assertRaisesRegex(SystemExit, "0"),
        ):
            main(["--help"])

        self.assertIn("usage: tensorwright", output.getvalue())
