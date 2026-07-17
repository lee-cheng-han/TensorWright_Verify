# Streaming and protocol diagnosis

Milestone 15 analyzes the candidate hardware stream independently from numerical diagnosis:

```bash
tensorwright trace diagnose-protocol reference.jsonl rtl.jsonl
tensorwright trace diagnose-protocol reference.jsonl rtl.jsonl --json
tensorwright trace diagnose-protocol reference.jsonl rtl.jsonl --report protocol.json
```

The versioned ruleset checks:

- every recorded transfer has `valid=true` and `ready=true`;
- accepted-transfer sequence numbers are zero-based and contiguous per operation and tensor;
- simulator cycles are present and strictly increasing;
- TLAST appears exactly on the final recorded transfer;
- semantic comparison contains neither missing nor unexpected candidate transfers.

Errors make the command exit with status 2. Missing optional diagnostic metadata is a warning.
Invalid traces or unreadable inputs exit with status 1. A clean protocol report exits with
status 0 even when values differ numerically; that mismatch belongs to `trace diagnose`.

Findings contain a stable rule ID, severity, event index, cycle when available, direct evidence,
and a recommended check. If duplicate semantic coordinates make comparison ambiguous, protocol
analysis reports that ambiguity instead of guessing an alignment.

The accepted-transfer trace cannot prove that output data remained stable during cycles where
`valid=true` and `ready=false`; that property remains enforced by SystemVerilog assertions and
would require cycle-sample trace events for offline diagnosis. The report therefore does not
claim stall-stability coverage from accepted transfers alone.
