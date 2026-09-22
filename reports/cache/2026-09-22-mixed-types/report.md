# OpenJev mixed-question timing: Qwen 0.8B

44 workloads × 3 repeats × 2 execution paths = 264 timed requests. The adaptive planner has a lower observed median on 44/44 workloads. The median of per-case speedups is 1.14×.

Both paths use the same loaded Qwen3.5-0.8B 4-bit weights, identical full prompts, and full-vocabulary P(yes). The previous multi-question path caches the context once, then batches independent question/candidate suffixes. The adaptive path also retains instructions from initialization and plans further shared prefixes. This compares execution speed and numerical agreement, not correctness or calibration.

## Short context, normal question and candidate lengths

N/C/S is the number of Noul/Choice/Score questions. Normal Choice and Score questions have three candidates each; Noul has one. Each request has one shared customer context. Timings are per-request medians, not per-question latency.

| Mix | N/C/S | Candidates | Previous | Adaptive | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced-3 | 1/1/1 | 7 | 181.5 ms | 146.0 ms | 1.24× |
| balanced-12 | 4/4/4 | 28 | 488.6 ms | 426.2 ms | 1.15× |
| balanced-24 | 8/8/8 | 56 | 966.5 ms | 810.4 ms | 1.19× |
| noul-heavy | 8/2/2 | 20 | 397.9 ms | 343.0 ms | 1.16× |
| choice-heavy | 2/8/2 | 32 | 567.1 ms | 494.0 ms | 1.15× |
| score-heavy | 2/2/8 | 32 | 543.1 ms | 477.9 ms | 1.14× |
| noul-only | 12/0/0 | 12 | 284.7 ms | 216.7 ms | 1.31× |
| choice-only | 0/12/0 | 36 | 613.6 ms | 517.5 ms | 1.19× |
| score-only | 0/0/12 | 36 | 604.3 ms | 501.6 ms | 1.20× |

## Context and suffix shape

The balanced 12-question workload contains four questions of each type. Long questions append repeated policy text to stress reusable question prefixes; uneven candidates deliberately lengthen one Choice/Score description. These are synthetic timing stressors. Wide options use five Choice/Score candidates.

| Context | Suffix shape | Candidates | Previous | Adaptive | Speedup | Previous/adaptive padding | Adaptive batches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| short | normal | 28 | 488.6 ms | 426.2 ms | 1.15× | 7/7 | [4, 4, 4, 4, 4, 4, 4] |
| short | long questions | 28 | 1385.9 ms | 911.1 ms | 1.52× | 7/20 | [4, 3, 3, 3, 3, 3, 3, 3, 3] |
| short | uneven candidates | 28 | 633.6 ms | 571.8 ms | 1.11× | 11/11 | [4, 4, 4, 4, 4, 4, 4] |
| long | normal | 28 | 709.9 ms | 664.2 ms | 1.07× | 7/7 | [4, 4, 4, 4, 4, 4, 4] |
| long | long questions | 28 | 1638.4 ms | 1146.5 ms | 1.43× | 7/20 | [4, 3, 3, 3, 3, 3, 3, 3, 3] |
| long | uneven candidates | 28 | 871.7 ms | 819.7 ms | 1.06× | 11/11 | [4, 4, 4, 4, 4, 4, 4] |
| short | wide options | 44 | 743.5 ms | 674.8 ms | 1.10× | 7/7 | [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] |
| long | wide options | 44 | 1016.5 ms | 936.5 ms | 1.09× | 7/7 | [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] |

## All measured cases

Min/max ranges cover 3 repeat measurements; they are not confidence intervals. All batch sizes are capped at four. Groups with different retained prefixes run sequentially; rows within each batch run in parallel. Question types share GPU batches but never share probability normalization.

| Workload | Previous median [min–max] ms | Adaptive median [min–max] ms | Speedup | Evaluated tokens previous/adaptive | Planner ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced-3-short-normal | 181.5 [169.2–182.5] | 146.0 [138.6–147.4] | 1.24× | 497/393 | 0.178 |
| balanced-3-short-long-questions | 418.0 [393.1–433.0] | 247.4 [244.4–260.8] | 1.69× | 1393/705 | 0.240 |
| balanced-3-short-uneven-candidates | 225.7 [224.4–240.2] | 198.2 [197.9–205.9] | 1.14× | 632/528 | 0.179 |
| balanced-3-long-normal | 328.8 [328.2–346.8] | 325.1 [306.8–332.2] | 1.01× | 1064/960 | 0.498 |
| balanced-3-long-long-questions | 560.9 [553.8–580.5] | 416.5 [414.5–432.3] | 1.35× | 1960/1272 | 0.485 |
| balanced-3-long-uneven-candidates | 394.7 [389.2–420.4] | 373.4 [372.7–385.2] | 1.06× | 1199/1095 | 0.425 |
| balanced-12-short-normal | 488.6 [483.6–531.3] | 426.2 [425.2–462.1] | 1.15× | 1505/1227 | 0.825 |
| balanced-12-short-long-questions | 1385.9 [1363.1–1431.8] | 911.1 [899.4–935.0] | 1.52× | 5089/2536 | 0.982 |
| balanced-12-short-uneven-candidates | 633.6 [630.4–672.4] | 571.8 [566.2–615.3] | 1.11× | 2045/1767 | 0.822 |
| balanced-12-long-normal | 709.9 [707.3–737.4] | 664.2 [663.7–709.1] | 1.07× | 2072/1794 | 2.224 |
| balanced-12-long-long-questions | 1638.4 [1624.3–1672.8] | 1146.5 [1124.1–1197.0] | 1.43× | 5656/3103 | 2.407 |
| balanced-12-long-uneven-candidates | 871.7 [843.1–887.9] | 819.7 [795.5–822.1] | 1.06× | 2612/2334 | 2.175 |
| balanced-24-short-normal | 966.5 [923.0–978.1] | 810.4 [794.0–814.8] | 1.19× | 2849/2268 | 1.752 |
| balanced-24-short-long-questions | 2703.0 [2662.7–2734.9] | 1813.4 [1802.6–1835.0] | 1.49× | 10017/4842 | 2.451 |
| balanced-24-short-uneven-candidates | 1205.8 [1197.3–1281.8] | 1119.7 [1101.0–1135.7] | 1.08× | 3929/3348 | 1.754 |
| balanced-24-long-normal | 1238.3 [1232.6–1300.0] | 1159.7 [1137.5–1167.9] | 1.07× | 3416/2835 | 5.147 |
| balanced-24-long-long-questions | 3047.7 [3021.0–3118.7] | 2166.9 [2119.3–2205.1] | 1.41× | 10584/5454 | 5.732 |
| balanced-24-long-uneven-candidates | 1532.6 [1515.3–1587.4] | 1418.6 [1404.3–1435.9] | 1.08× | 4496/3915 | 5.155 |
| noul-heavy-short-normal | 397.9 [380.9–411.5] | 343.0 [331.6–349.9] | 1.16× | 1200/940 | 0.769 |
| noul-heavy-short-long-questions | 1022.8 [1002.5–1058.8] | 793.1 [748.0–802.8] | 1.29× | 3760/2330 | 0.788 |
| noul-heavy-short-uneven-candidates | 502.2 [482.8–513.5] | 449.2 [440.9–451.7] | 1.12× | 1470/1210 | 0.741 |
| noul-heavy-long-normal | 611.9 [608.2–626.6] | 573.4 [552.4–575.8] | 1.07× | 1767/1507 | 1.685 |
| noul-heavy-long-long-questions | 1261.2 [1247.2–1321.6] | 982.1 [972.3–1034.2] | 1.28× | 4327/2897 | 1.797 |
| noul-heavy-long-uneven-candidates | 694.6 [670.5–706.7] | 650.1 [629.1–670.5] | 1.07× | 2037/1777 | 1.644 |
| choice-heavy-short-normal | 567.1 [561.5–589.6] | 494.0 [493.4–523.3] | 1.15× | 1651/1422 | 1.180 |
| choice-heavy-short-long-questions | 1572.4 [1558.6–1603.0] | 974.1 [968.4–1005.1] | 1.61× | 5747/2592 | 1.433 |
| choice-heavy-short-uneven-candidates | 786.6 [759.0–803.5] | 709.3 [705.0–718.0] | 1.11× | 2461/2232 | 1.360 |
| choice-heavy-long-normal | 776.8 [772.8–816.6] | 730.4 [718.5–771.4] | 1.06× | 2218/1989 | 2.921 |
| choice-heavy-long-long-questions | 1808.4 [1805.4–1839.8] | 1259.6 [1211.1–1280.4] | 1.44× | 6314/3159 | 3.431 |
| choice-heavy-long-uneven-candidates | 993.8 [992.6–1035.8] | 947.9 [939.5–969.0] | 1.05× | 3028/2799 | 2.936 |
| score-heavy-short-normal | 543.1 [540.0–546.9] | 477.9 [469.1–501.4] | 1.14× | 1658/1325 | 1.024 |
| score-heavy-short-long-questions | 1565.1 [1548.3–1567.7] | 971.5 [959.5–991.9] | 1.61× | 5754/2591 | 1.392 |
| score-heavy-short-uneven-candidates | 709.8 [708.9–760.3] | 606.4 [604.9–641.9] | 1.17× | 2198/1882 | 1.157 |
| score-heavy-long-normal | 794.2 [772.8–814.2] | 729.2 [720.9–742.4] | 1.09× | 2225/1996 | 3.039 |
| score-heavy-long-long-questions | 1820.2 [1794.1–1880.5] | 1222.6 [1215.0–1222.7] | 1.49× | 6321/3158 | 3.320 |
| score-heavy-long-uneven-candidates | 995.8 [955.3–999.5] | 883.7 [855.3–924.2] | 1.13× | 2765/2449 | 2.927 |
| noul-only-short-normal | 284.7 [282.0–304.9] | 216.7 [215.7–232.5] | 1.31× | 904/632 | 0.376 |
| noul-only-long-normal | 477.1 [467.1–489.7] | 434.8 [413.3–437.1] | 1.10× | 1471/1199 | 0.873 |
| choice-only-short-normal | 613.6 [592.4–656.9] | 517.5 [503.6–542.6] | 1.19× | 1796/1442 | 0.906 |
| choice-only-long-normal | 839.0 [829.6–862.5] | 796.9 [774.3–815.7] | 1.05× | 2363/2009 | 2.522 |
| score-only-short-normal | 604.3 [583.9–605.9] | 501.6 [501.0–527.9] | 1.20× | 1823/1434 | 0.645 |
| score-only-long-normal | 828.7 [821.7–854.4] | 765.3 [743.3–776.9] | 1.08× | 2390/2001 | 1.947 |
| balanced-12-short-wide-options | 743.5 [735.2–793.4] | 674.8 [639.3–676.0] | 1.10× | 2273/1867 | 1.083 |
| balanced-12-long-wide-options | 1016.5 [990.3–1023.8] | 936.5 [912.9–957.9] | 1.09× | 2840/2434 | 3.680 |

## Output differences and verification

Distinct question instances: 546. Choice selected-answer agreement: 169/182. Noul mean/max absolute output difference: 0.009824/0.031764. Score mean/max absolute expected-index difference: 0.008267/0.029684 (0–2 normally; 0–4 with five levels).

Quantized execution shapes can change scores. Agreement does not measure accuracy. Each output was recomputed from only that question's candidate supports: Noul equals its support, Choice normalizes its own alternatives, and Score is the expected level under its own normalized distribution. Repeated scores identical within each path: True.

The untimed audit reconstructed exact original prompt tokens for 88 requests and checked 376 retained parent-cache uses. All checked KV/recurrent caches remained unchanged. The permanent cache remained unchanged after the complete timed run.

## Protocol

Model `mlx-community/Qwen3.5-0.8B-4bit`, revision `da28692b5f139cb0ec58a356b437486b7dac7462`. Runtime `{'mlx': '0.32.2', 'mlx-lm': '0.32.0', 'transformers': '5.17.0'}`, platform `macOS-27.0-arm64-arm-64bit`. All model layers are retained; thinking is disabled and no tokens are generated. The scorer and planner are unchanged from the prior cache benchmark.

Both methods warm up on every case, then alternate execution order. Timing includes prompt compilation, planning, prefill, cache copying, scoring, result construction and GPU synchronization. Warmup and cache-audit hashing are excluded. Model loading is excluded; instruction-cache initialization is reported separately: 74 permanent tokens took 94.45 ms to initialize after model load. Reuse counts exclude this one-time computation. The planner uses heuristic cost estimates, not a guarantee of globally optimal batching.

[Inputs](cases.json) · [Every timed call](calls.jsonl) · [Per-case and per-type summary](mixed-summary.json) · [Timing summary](summary.json) · [Protocol and source hashes](protocol.json) · [Cache audit](verification.json) · [Initialization](initialization.json)
