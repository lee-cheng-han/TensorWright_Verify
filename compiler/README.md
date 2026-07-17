# Compiler staging area

The bit-accurate integer reference lives in `tensorwright.reference` so it can be shared
by compiler and verification code. The Milestone 2 ONNX frontend and typed IR live in
`tensorwright.compiler`, alongside the Milestone 3 passes in
`tensorwright.compiler.passes`. This top-level directory records the architectural
subsystem boundary without duplicating the importable Python package.
