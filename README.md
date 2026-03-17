# regmixer

Round1a 的训练、评测、拟合和可视化入口是 `scripts/run_round1a.sh`。

Round1a 的 mix 生成现在明确拆成两层：

- candidate sampling：默认兼容旧的 `dirichlet`，新的 round1a 配置使用 `mixed`
- design selection：默认兼容旧的 `random`，新的 round1a 配置使用 `d_opt`

`generate-mixes` 会同时写：

- `*_mixes.json`
- `*_design_summary.json`
- `design/` 诊断图目录

Round1a 的 canonical 拟合默认使用 `log_linear`。

回归拟合的使用方法、回归器选择建议，以及 `ROUND1A_FIT_REGRESSION_TYPE` /
`ROUND1A_COMPARE_REGRESSIONS` 的接入方式，见 [DOCS.md](DOCS.md) 中的
“Round1a 拟合与可视化”一节。
