"""按 V1 原模型重训七组，并运行 2024--2026 有状态执行约束回测。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.features import (
    build_group_matrix,
    channel_ids,
    load_feature_matrices,
)
from src.models.cross_sectional_lightgbm import (
    date_equal_sample_weights,
    date_membership_mask,
    rank_gaussian_target_by_date,
)
from src.models.cross_sectional_lightgbm_final import (
    build_walk_forward_folds_range,
    fit_lightgbm_final,
)
from src.portfolio.execution_constraints import (
    simulate_topk_open_execution,
    summarize_execution_backtest,
)


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    """计算单日 Spearman RankIC；标签缺失行只影响评价，不影响预测排名。"""

    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan
    return float(x.loc[valid].corr(y.loc[valid], method="spearman"))


def _prediction_summary(predictions: pd.DataFrame) -> dict[str, float]:
    """汇总只在 label 可观测行上定义的预测指标。"""

    daily_ic = predictions.groupby("date", observed=True).apply(
        lambda block: _rank_ic(block["prediction"], block["label_1d"])
    )
    daily_ic = daily_ic.dropna()
    standard_deviation = float(daily_ic.std(ddof=1))
    return {
        "mean_rank_ic": float(daily_ic.mean()),
        "rank_icir": (
            float(daily_ic.mean() / standard_deviation)
            if standard_deviation > 0
            else np.nan
        ),
        "rank_ic_positive_ratio": float(daily_ic.gt(0).mean()),
        "prediction_rows": float(len(predictions)),
        "prediction_rows_without_label": float(predictions["label_1d"].isna().sum()),
    }


def _period_tables(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成执行约束后的逐年和逐月结果。"""

    yearly_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for group, group_block in daily.groupby("feature_group", sort=False):
        for year, block in group_block.groupby(group_block["date"].dt.year):
            summary = summarize_execution_backtest(block)
            summary["annual_excess_return"] = float(
                (block["portfolio_return"] - block["benchmark_return"]).mean()
                * 252.0
            )
            yearly_rows.append(
                {"feature_group": group, "year": int(year), **summary}
            )
        month_key = group_block["date"].dt.to_period("M").astype(str)
        for month, block in group_block.groupby(month_key):
            summary = summarize_execution_backtest(block)
            summary["period_excess_return"] = float(
                (1.0 + block["portfolio_return"] - block["benchmark_return"]).prod()
                - 1.0
            )
            monthly_rows.append(
                {"feature_group": group, "month": str(month), **summary}
            )
    return pd.DataFrame(yearly_rows), pd.DataFrame(monthly_rows)


def _top100(predictions: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """从完整 OOS 截面选 Top-100，绝不按未来 label 或执行状态预筛。"""

    ordered = predictions.sort_values(
        ["date", "prediction", "stock"], ascending=[True, False, True]
    )
    selected = ordered.groupby("date", sort=True, observed=True).head(top_k).copy()
    selected["rank"] = selected.groupby("date", observed=True).cumcount() + 1
    counts = selected.groupby("date", observed=True)["stock"].size()
    if not counts.eq(top_k).all():
        raise ValueError(f"存在不足 Top-{top_k} 的 OOS 日期")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lightgbm_final_v11.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "lightgbm_final_v11" / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "lightgbm_final_v11" / "results",
    )
    args = parser.parse_args()
    import lightgbm as lgb

    started = perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    features = config["features"]
    validation = config["validation"]
    model_config = config["model"]
    portfolio = config["portfolio"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "frozen_config.json")

    representatives_path = PROJECT_ROOT / str(data["candidate_representatives"])
    (
        base,
        traditional,
        traditional_names,
        gp,
        gp_names,
        representatives,
    ) = load_feature_matrices(
        args.data_dir,
        representatives_path,
        expected_traditional=int(features["traditional_count"]),
        expected_gp=int(features["union_gp_count"]),
    )
    if pd.Timestamp(base["date"].max()) != pd.Timestamp(data["final_data_end"]):
        raise ValueError("V1.1 准备数据未覆盖到 2026-08-19")

    target = rank_gaussian_target_by_date(base["date"], base["label_1d"])
    valid_target = np.isfinite(target)
    sample_weight = date_equal_sample_weights(base["date"], valid_target)
    rankic_ids, mi_ids = channel_ids(
        representatives,
        expected_rankic=int(features["rankic_gp_count"]),
        expected_mi=int(features["mi_gp_count"]),
    )
    folds = build_walk_forward_folds_range(
        base.loc[valid_target, "date"].drop_duplicates(),
        oos_start=str(data["oos_start"]),
        oos_end=str(data["oos_end"]),
        rolling_train_dates=int(validation["rolling_train_dates"]),
        inner_validation_dates=int(validation["inner_validation_dates"]),
        inner_gap_dates=int(validation["inner_gap_dates"]),
        outer_gap_dates=int(validation["outer_gap_dates"]),
        test_dates=int(validation["test_dates"]),
        step_dates=int(validation["step_dates"]),
    )

    parameters = {
        key: model_config[key]
        for key in (
            "objective",
            "metric",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_data_in_leaf",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "max_bin",
            "seed",
            "num_threads",
        )
    }
    parameters.update(
        {
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
            "feature_fraction_seed": int(model_config["seed"]),
            "bagging_seed": int(model_config["seed"]),
            "data_random_seed": int(model_config["seed"]),
        }
    )

    execution_bars = pd.read_parquet(PROJECT_ROOT / str(data["execution_bars"]))
    calendar_frame = pd.read_csv(PROJECT_ROOT / str(data["trading_calendar"]))
    trading_calendar = pd.DatetimeIndex(
        pd.to_datetime(
            calendar_frame.loc[
                calendar_frame["is_trading_day"].astype(str).eq("1"),
                "calendar_date",
            ]
        )
    )
    benchmark = (
        base.loc[
            base["date"].between(data["oos_start"], data["oos_end"]),
            ["date", "label_1d"],
        ]
        .groupby("date", as_index=False, observed=True)["label_1d"]
        .mean()
        .rename(columns={"label_1d": "benchmark_return"})
    )

    date_array = base["date"].to_numpy()
    scoreboard_rows: list[dict[str, object]] = []
    all_daily: list[pd.DataFrame] = []
    for group_value in features["groups"]:
        group = str(group_value)
        matrix, feature_names = build_group_matrix(
            group,
            traditional,
            traditional_names,
            gp,
            gp_names,
            rankic_ids=rankic_ids,
            mi_ids=mi_ids,
            cmi_core=[str(value) for value in features["cmi_core_ids"]],
        )
        group_dir = args.output_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        fold_audit: list[dict[str, object]] = []
        importance_rows: list[pd.DataFrame] = []

        for fold in folds:
            stem = f"fold_{fold.fold:03d}"
            prediction_path = group_dir / f"{stem}_predictions.parquet"
            audit_path = group_dir / f"{stem}_audit.json"
            importance_path = group_dir / f"{stem}_importance.csv"
            if prediction_path.exists() and audit_path.exists() and importance_path.exists():
                fold_audit.append(json.loads(audit_path.read_text(encoding="utf-8")))
                importance_rows.append(pd.read_csv(importance_path))
                continue

            inner_train_mask = valid_target & date_membership_mask(
                date_array, fold.inner_train_dates
            )
            inner_valid_mask = valid_target & date_membership_mask(
                date_array, fold.inner_validation_dates
            )
            final_train_mask = valid_target & date_membership_mask(
                date_array, fold.final_train_dates
            )
            # V1.1 的核心修复：OOS 预测不与 valid_target 相交。未来停牌或
            # 卖出受阻导致 label 缺失的股票仍必须参与当日完整截面排名。
            test_predict_mask = date_membership_mask(date_array, fold.test_dates)
            row_counts = {
                "inner_train_rows": int(inner_train_mask.sum()),
                "inner_validation_rows": int(inner_valid_mask.sum()),
                "final_train_rows": int(final_train_mask.sum()),
                "test_prediction_rows": int(test_predict_mask.sum()),
                "test_labeled_rows": int((test_predict_mask & valid_target).sum()),
                "test_unlabeled_rows_retained": int(
                    (test_predict_mask & ~valid_target).sum()
                ),
            }
            if min(
                row_counts["inner_train_rows"],
                row_counts["inner_validation_rows"],
                row_counts["final_train_rows"],
                row_counts["test_prediction_rows"],
            ) <= 0:
                raise ValueError(f"{group} fold {fold.fold} 存在空窗口: {row_counts}")
            print(
                f"{group} fold {fold.fold}/{len(folds)} rows={row_counts}",
                flush=True,
            )
            prediction, best_iteration, history, gain, split = fit_lightgbm_final(
                matrix[inner_train_mask],
                target[inner_train_mask],
                sample_weight[inner_train_mask],
                matrix[inner_valid_mask],
                target[inner_valid_mask],
                sample_weight[inner_valid_mask],
                matrix[final_train_mask],
                target[final_train_mask],
                sample_weight[final_train_mask],
                matrix[test_predict_mask],
                feature_names=feature_names,
                parameters=parameters,
                num_boost_round=int(model_config["num_boost_round"]),
                early_stopping_rounds=int(model_config["early_stopping_rounds"]),
            )
            prediction_frame = base.loc[
                test_predict_mask, ["date", "stock", "label_1d"]
            ].copy()
            prediction_frame["target_rankgauss"] = target[test_predict_mask]
            prediction_frame["prediction"] = prediction
            prediction_frame["feature_group"] = group
            prediction_frame["fold"] = fold.fold
            prediction_frame.to_parquet(
                prediction_path, index=False, compression="zstd"
            )

            gain_total = float(gain.sum())
            split_total = float(split.sum())
            importance = pd.DataFrame(
                {
                    "feature_group": group,
                    "fold": fold.fold,
                    "feature": feature_names,
                    "gain": gain,
                    "gain_share": gain / gain_total if gain_total > 0 else 0.0,
                    "split": split,
                    "split_share": split / split_total if split_total > 0 else 0.0,
                }
            )
            importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
            importance_rows.append(importance)
            record = {
                "feature_group": group,
                "fold": fold.fold,
                "inner_train_start": str(min(fold.inner_train_dates).date()),
                "inner_train_end": str(max(fold.inner_train_dates).date()),
                "inner_validation_start": str(min(fold.inner_validation_dates).date()),
                "inner_validation_end": str(max(fold.inner_validation_dates).date()),
                "final_train_start": str(min(fold.final_train_dates).date()),
                "final_train_end": str(max(fold.final_train_dates).date()),
                "test_start": str(min(fold.test_dates).date()),
                "test_end": str(max(fold.test_dates).date()),
                "best_iteration": best_iteration,
                "feature_count": int(matrix.shape[1]),
                **row_counts,
                "final_validation_l2": float(history.get("l2", [np.nan])[-1]),
            }
            audit_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            fold_audit.append(record)
            print(
                f"{group} fold {fold.fold}/{len(folds)} complete, "
                f"best_iteration={best_iteration}",
                flush=True,
            )

        prediction_files = sorted(group_dir.glob("fold_*_predictions.parquet"))
        if len(prediction_files) != len(folds):
            raise ValueError(f"{group} OOS prediction fold 数量不完整")
        predictions = pd.concat(
            [pd.read_parquet(path) for path in prediction_files], ignore_index=True
        ).sort_values(["date", "stock"]).reset_index(drop=True)
        if predictions.duplicated(["date", "stock"]).any():
            raise ValueError(f"{group} OOS 预测键重复")
        predictions.to_parquet(
            group_dir / "oos_predictions.parquet", index=False, compression="zstd"
        )

        planned = _top100(predictions, int(portfolio["top_k"]))
        planned.to_parquet(
            group_dir / "planned_top100.parquet", index=False, compression="zstd"
        )
        execution = simulate_topk_open_execution(
            planned[["date", "stock", "rank"]],
            execution_bars,
            trading_calendar,
            top_k=int(portfolio["top_k"]),
        )
        daily = execution.daily.merge(
            benchmark,
            left_on="signal_date",
            right_on="date",
            how="left",
            validate="one_to_one",
        )
        if daily["benchmark_return"].isna().any():
            raise ValueError(f"{group} 基准收益无法与执行日期对齐")
        daily["feature_group"] = group
        daily["excess_return"] = daily["portfolio_return"] - daily["benchmark_return"]
        daily.to_csv(group_dir / "daily_execution_metrics.csv", index=False)
        events = execution.events.copy()
        events["feature_group"] = group
        events.to_csv(group_dir / "execution_events.csv", index=False)
        terminal = execution.terminal_positions.copy()
        terminal["feature_group"] = group
        terminal.to_csv(group_dir / "terminal_positions.csv", index=False)

        importance_all = pd.concat(importance_rows, ignore_index=True)
        importance_summary = (
            importance_all.groupby("feature", as_index=False)
            .agg(
                mean_gain_share=("gain_share", "mean"),
                mean_split_share=("split_share", "mean"),
                folds_with_positive_gain=("gain", lambda x: int((x > 0).sum())),
            )
            .sort_values("mean_gain_share", ascending=False)
        )
        importance_summary.insert(0, "feature_group", group)
        importance_summary.to_csv(
            group_dir / "feature_importance_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary: dict[str, object] = {
            "feature_group": group,
            "feature_count": int(matrix.shape[1]),
            **summarize_execution_backtest(daily),
            **_prediction_summary(predictions),
            "annual_excess_return": float(daily["excess_return"].mean() * 252.0),
            "mean_best_iteration": float(
                pd.DataFrame(fold_audit)["best_iteration"].mean()
            ),
            "total_blocked_buy_slots": int(daily["n_blocked_buy"].sum()),
            "total_blocked_sell_attempts": int(daily["n_blocked_sell"].sum()),
            "terminal_unsellable_count": int(
                (~terminal["can_sell_at_final_exit"].astype(bool)).sum()
            ),
        }
        # 修复后的 Top-100 必须真实遇到至少一种约束；全部为零说明未来筛选复发。
        if (
            summary["total_blocked_buy_slots"] == 0
            and summary["total_blocked_sell_attempts"] == 0
        ):
            raise ValueError(f"{group} 未观察到任何执行约束，拒绝输出结果")
        scoreboard_rows.append(summary)
        all_daily.append(daily)
        del matrix, predictions, planned, execution, daily, importance_all
        gc.collect()

    scoreboard = pd.DataFrame(scoreboard_rows).sort_values(
        "annual_return", ascending=False
    )
    scoreboard.to_csv(args.output_dir / "scoreboard.csv", index=False)
    daily_all = pd.concat(all_daily, ignore_index=True)
    daily_all.to_csv(args.output_dir / "all_daily_execution_metrics.csv", index=False)
    yearly, monthly = _period_tables(daily_all)
    yearly.to_csv(args.output_dir / "yearly_scoreboard.csv", index=False)
    monthly.to_csv(args.output_dir / "monthly_scoreboard.csv", index=False)
    stress_month = str(config["reporting"]["stress_month"])
    stress = monthly.loc[monthly["month"].eq(stress_month)].copy()
    if len(stress) != len(features["groups"]):
        raise ValueError(f"压力月份 {stress_month} 的七组结果不完整")
    stress.to_csv(args.output_dir / f"stress_month_{stress_month}.csv", index=False)

    run_manifest = {
        "status": "complete",
        "research_version": config["research_version"],
        "reference_model": "LightGBM Final V1",
        "v1_features_validation_model_parameters_unchanged": True,
        "repair_scope": config["repair_scope"],
        "oos_start": data["oos_start"],
        "oos_end": data["oos_end"],
        "fold_count": len(folds),
        "feature_groups": list(features["groups"]),
        "top_k": int(portfolio["top_k"]),
        "transaction_cost_bps": float(portfolio["round_trip_cost_bps"]),
        "execution_mode": portfolio["execution_mode"],
        "full_cross_section_predictions_saved": True,
        "label_missing_oos_rows_were_predicted": True,
        "original_locked_holdout_previously_opened": True,
        "this_run_is_new_locked_holdout": False,
        "interpretation": "V1.1 data-contract repair sensitivity run",
        "lightgbm_version": lgb.__version__,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(scoreboard.to_string(index=False), flush=True)
    print("\nYEARLY\n", yearly.to_string(index=False), flush=True)
    print(f"\nSTRESS MONTH {stress_month}\n", stress.to_string(index=False), flush=True)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
