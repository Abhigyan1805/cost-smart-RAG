| model | route | coverage (95% CI) | C4-gap | McNemar p | GPU-s/query | cost ratio r | saving @ gate | verdict |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | L0 | 0.095 [0.055, 0.140] | 0.905 | 7.99e-41 | 1.00 | 0.1656 | 0.0834 | NO-GO |
| Qwen2.5-1.5B-Instruct | L1 | 0.125 [0.080, 0.170] | 0.875 | 1.63e-39 | 3.25 | 0.5364 | 0.0464 | GO |
| Qwen2.5-3B-Instruct | L0 | 0.155 [0.105, 0.205] | 0.845 | 3.34e-38 | 3.27 | 0.5399 | 0.0460 | GO |
| Qwen2.5-3B-Instruct | L1 | 0.085 [0.050, 0.125] | 0.915 | 2.92e-41 | 6.80 | 1.1216 | -0.0122 | NO-GO |
| Qwen2.5-3B-Instruct | C0 | 0.170 [0.120, 0.225] | 0.830 | 1.51e-37 | 5.07 | 0.8364 | 0.0164 | GO |
| Qwen2.5-7B-Instruct | L0 | 0.150 [0.100, 0.200] | 0.850 | 2.02e-38 | 1.86 | 0.3060 | 0.0694 | GO |
| Qwen2.5-7B-Instruct | L1 | 0.065 [0.035, 0.100] | 0.935 | 3.91e-42 | 8.03 | 1.3246 | -0.0325 | NO-GO |
| Qwen2.5-7B-Instruct | C0 | 0.075 [0.040, 0.115] | 0.925 | 1.07e-41 | 13.74 | 2.2654 | -0.1265 | NO-GO |

Routing pays when cheap-tier coverage exceeds the 0.10 exploitable-separation gate. The closed-book cheap route L0 first clears the gate at Qwen/Qwen2.5-3B-Instruct (coverage 0.155 [0.105, 0.205], cost ratio r=0.5399).

## Provenance

- Qwen/Qwen2.5-1.5B-Instruct: sweep `results/realdata-15/sweep.db`, repeats `results/realdata-15/repeats.db`; attestation server-attested=0, unattested=1600, stub=0
- Qwen/Qwen2.5-3B-Instruct: sweep `results/tiersweep-16/sweep-3b.db`, repeats `results/tiersweep-16/repeats-3b.db`; attestation server-attested=2400, unattested=0, stub=0
- Qwen/Qwen2.5-7B-Instruct: sweep `results/tiersweep-16/sweep-7b.db`, repeats `results/tiersweep-16/repeats-7b.db`; attestation server-attested=2400, unattested=0, stub=0
