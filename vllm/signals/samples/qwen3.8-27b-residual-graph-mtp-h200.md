# qwen3.8-27b-residual-graph-mtp-h200.safetensors

**The verified configuration.** Qwen3.8-27B-INT4 on an H200 NVL, captured with
`--signal-capture-backend graph` while MTP speculative decoding was running and
CUDA graphs were compiled. No forward hooks, no eager execution.

This is the deposit the whole patch exists to produce.

| field | value |
|---|---|
| model | `/home/ubuntu/models/RedHatAI/Qwen3.8-27B-INT4` |
| tier | `residual_raw` |
| token_reduce | `last` |
| num_captured_positions | `85` |
| dtypes | `bfloat16,float32` |
| format | `sigcap-v1` |

| tensor | shape | dtype | bytes |
|---|---|---|---|
| `logit` | [1, 3] | F32 | 12 |
| `logit.index` | [1, 1] | F32 | 4 |
| `residual` | [1, 5120] | BF16 | 10,240 |
| `residual.index` | [1, 2] | F32 | 8 |

## The residual

- layer **63** of 64 (the last)
- **5120 dims, BF16 = 10,240 bytes** exactly
- L2 norm **349.89**, mean abs 3.621, max abs 61.5
- all values finite: **True**

`num_captured_positions` exceeds the turn's completion tokens because MTP
verifies draft positions - capture sees each verified position, not just the
tokens finally emitted.

## First 16 values

```
-4.5312  -3.3125  -1.4688  -0.0859  -10.1875  -4.0938  +4.3125  -0.6328  +7.3125  -2.4844  -0.2891  +2.2500  +7.4062  -2.1094  -6.6250  +1.6875
```

