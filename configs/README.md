# configs

公开版本只保留两份最终配置：

- `large_scale_gp_v2.json`：RankIC/MI 双通道岛屿 GP、语义多样性和开发期置换检验；
- `lightgbm_final_v11.json`：七组因子对照、固定滚动 walk-forward、Top-100 与状态化执行。

配置全部使用仓库相对路径。原始行情日期由数据入口的命令行参数指定；冻结结果所用日期记录在 LightGBM 配置与结果 manifest 中。
