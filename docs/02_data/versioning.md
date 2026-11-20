# versioning.md —— 版本、复现与监控（D7）

> 生成：2026-11-20 06:04

## 一、数据快照

| 项 | 值 |
|---|---|
| **snapshotId** | `snap-1778b81371ea59fa` |
| 文件数 | 227 |
| 总字节 | 11.32 GB |
| 计算时间 | 2026-11-20 06:04 |

**哈希算法**：sha256( 每个文件的 `相对路径|字节数|mtime` 按序拼接 )
任何文件内容或修改时间变化，哈希即变。

## 二、规则版本

| 规则 | 版本 | 生效日 | 内容 |
|---|---|---|---|
| universe | **v1.0** | 2026-11-20 | `{"min_listed_days": 250, "exclude_st": true, "amount_rank_min": 0.2, "require_quote": true}` |
| mask | **v1.0** | 2026-11-20 | `{"exclude_limit_up": true, "exclude_limit_down": false}` |
| label | **v1.0** | 2026-11-20 | `{"convention": "next-open-to-open", "horizon": 60, "fill_price": "open", "rebalance_interval": 20}` |

## 三、代码版本

| 脚本 | sha256 前 16 位 |
|---|---|
| `CLEAN.py` | `96bb8e70c031b763` |
| `D2.py` | `3087d57c1796106e` |
| `build_fin_factors.py` | `39458e19ec41d0b1` |
| `clean_all.py` | `ac86ccb77470b6e6` |
| `clean_fin.py` | `cd1c8af4d921620a` |
| `d2_pit.py` | `2918a3c552ad7859` |
| `d3_universe.py` | `34b211f81011c73d` |
| `d4_label.py` | `d9efd353398f692d` |
| `d5_split.py` | `c9b89fe7513c86bf` |
| `d6_missing.py` | `98bcc7d1dcc03af5` |
| `d7_versioning.py` | `43409d2e4a75ebdd` |
| `export_mlft_data.py` | `9e422035070c9c50` |
| `label.py` | `ce3dbcc0d9df281b` |

## 四、参数清单

| 参数 | 值 |
|---|---|
| `H` | 60 |
| `purge_calendar_days` | 90 |
| `cost_bp_one_way` | 10 |
| `mad_k` | 3 |
| `mad_to_sigma` | 1.4826 |
| `winsor_p` | 0.01 |
| `train_end` | 2024-09-30 |
| `holdout_start` | 2025-01-01 |
| `n_splits` | 5 |
| `min_listed_days` | 250 |

## 五、随机种子

| 用途 | seed |
|---|---|
| random_forest | 42 |
| permutation_test | 42 |
| gbdt | 42 |

## 六、运行顺序

```
 1. python clean_all.py
 2. python clean_fin.py
 3. python build_fin_factors.py
 4. python export_mlft_data.py
 5. python d2_pit.py
 6. python d3_universe.py
 7. python d4_label.py
 8. python d5_split.py
 9. python d6_missing.py
10. python d7_versioning.py
```

每一步的输入是上一步的输出；全部脚本幂等，可单独重跑。

## 七、PSI 监控

| 因子 | 基准期 | 当前期 | PSI | 判定 |
|---|---|---|---|---|
| `gm` | 训练窗 | 测试窗 | 0.0314 | 稳定 |
| `dgm_q` | 训练窗 | 测试窗 | 0.4927 | 重训 |
| `transfer_intensity` | 训练窗 | 测试窗 | 0.0455 | 稳定 |
| `dep_rate` | 训练窗 | 测试窗 | 0.0231 | 稳定 |
| `gm_vol` | 训练窗 | 测试窗 | 0.0090 | 稳定 |

**阈值**：< 0.10 稳定 ｜ 0.10~0.25 关注 ｜ > 0.25 重训

**纪律**：PSI 必须与 IC 同时看——
PSI 小但 IC 降 = alpha 衰减（降权/退役）；
PSI 大但 IC 稳 = 分布漂移但仍有效（观察）。


## 八、复现检查清单

- [x] 1. 数据快照 ID 已固定
- [x] 2. 三条规则版本已固定
- [x] 3. 脚本哈希已记录
- [x] 4. 参数全部在配置中
- [x] 5. 随机种子已记录
- [x] 6. 运行顺序已编号