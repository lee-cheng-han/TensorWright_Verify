# Compiler staging area

The bit-accurate integer reference lives in `tensorwright.reference` so it can be shared
by compiler and verification code. The Milestone 2 ONNX frontend and typed IR live in
`tensorwright.compiler`. This top-level directory records the architectural subsystem
boundary; future compiler passes may move beneath it once they contain functional code.
