# qwen3-0.6b-residual-graph-backend.safetensors

Real capture from a live vLLM run. Raw binary is the `.safetensors`;
this file is the same data rendered for reading.

## Deposit

| field | value |
|---|---|
| model | `Qwen/Qwen3-0.6B` |
| tier | `residual_raw` |
| token_reduce | `last` |
| num_captured_positions | `24` |
| dtypes | `bfloat16,float32` |
| engine | `vllm` |
| format | `sigcap-v1` |

## Tensors

| tensor | shape | dtype | bytes |
|---|---|---|---|
| `logit` | [1, 3] | F32 | 12 |
| `logit.index` | [1, 1] | F32 | 4 |
| `residual` | [1, 1024] | BF16 | 2,048 |
| `residual.index` | [1, 2] | F32 | 8 |

## Residual: 1 row(s) x 1024 dims

| layer | L2 norm | cos(prev layer) | mean abs | max abs |
|---|---|---|---|---|
| 27 | 626.93 | 0.0000 | 9.842 | 308.0 |

## Last row, first 16 values

```
+0.6406  -11.6250  +228.0000  +53.0000  +14.9375  -76.0000  -8.4375  -108.0000  -19.0000  +7.8125  -15.0000  -0.9531  +28.8750  +308.0000  -13.8750  +14.6875
```

