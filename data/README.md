# data

```text
raw/        可选的供应商原始文件
interim/    可删除、可重建的单股分片和临时面板
processed/  正式训练输入
```

真实行情与宽因子表体积较大且受数据许可约束，因此不提交 Git。三个目录只保留占位文件。

正式数据入口是 `scripts/build_lightgbm_final_v11_data.py`，输出：

```text
processed/clean_panel_v11.parquet
processed/execution_bars_v11.parquet
processed/trading_calendar_v11.csv
processed/data_summary_v11.json
```

`clean_panel_v11` 保存 `(date, stock)`、OHLCV、换手率、上市交易日数、1 日 Open-to-Open 标签和未来执行标记。未来执行标记不参与信号日样本筛选。

行业中性化还需要用户提供 Point-in-Time 月度行业映射：

```text
processed/industry_monthly_pit.csv
```

字段及完整复现命令见 `docs/REPRODUCIBILITY.md`。
