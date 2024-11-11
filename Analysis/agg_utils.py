import numpy as np
import polars as pl
from typing import Callable
from numpy.typing import NDArray
from scipy.stats import bootstrap

def agg_case(df: pl.DataFrame) -> pl.DataFrame:
    """
    aggregate case level prediction
    predict as {}, {0}, {1}, {0, 1} 
    if one of these prediction sets occupies more than 50% of the cases.
    Otherwise, predict as ambiguous
    """
    agg_df = df \
        .group_by("case") \
        .agg(
            pl.col("label").first(),
            pl.col("ihc_score").first(),
            (pl.col("final_pred") == 0).sum().alias("fpred0"),
            (pl.col("final_pred") == 1).sum().alias("fpred1"),
            ((pl.col("final_pred") == -1) & (pl.col("pred_size") == 2)).sum().alias("fpred2"),
            ((pl.col("final_pred") == -1) & (pl.col("pred_size") == 0)).sum().alias("fpred-1"),
            pl.len().alias("count")
        ) \
        .with_columns( # calculate final_pred for case
            pl
            .when(pl.col("fpred0") / pl.col("count") > 0.5).then(pl.lit(0)) # {0}
            .when(pl.col("fpred1") / pl.col("count") > 0.5).then(pl.lit(1)) # {1}
            .when(pl.col("fpred-1") / pl.col("count") > 0.5).then(pl.lit(-1)) # {}
            .otherwise(pl.lit(-1)) # -1 with pred_size = 2 {0, 1}
            .alias("final_pred")
        ) \
        .with_columns( # calculate final_pred for case
            pl
            .when(pl.col("fpred0") / pl.col("count") > 0.5).then(pl.lit(1)) # {0}
            .when(pl.col("fpred1") / pl.col("count") > 0.5).then(pl.lit(1)) # {1}
            .when(pl.col("fpred-1") / pl.col("count") > 0.5).then(pl.lit(0)) # {}
            .otherwise(pl.lit(2)) # -1 with pred_size = 2 {0, 1}
            .alias("pred_size")
        ) 
    return agg_df
    
def agg_heights(
        root: str, 
        r_min: int, 
        r_max: int, 
        alphas: NDArray, 
        agg_func: Callable[[pl.DataFrame], list], 
        col_names: list
    ) -> dict:
    """
    aggregate avg. (heights) based on the aggregated function
    from multiple experiments to approaximate CI
    """
    rows = []
    for r in range(r_min, r_max + 1):
        # fold which is 4 (remember we split 1/5 for calibration)
        for f in range(4):
            for alpha in alphas:
                # we consider only the case alpha0=alpha1 in the scope of our work
                file_path = root / f"{r}_{f}" / f"{r}_{f}_alpha0_{alpha}_alpha1_{alpha}_result.csv"

                df = pl.read_csv(file_path)
                values = agg_func(df)

                row = [r, alpha] + values
                rows.append(row)
    
    schema = ["r", "alpha"] + col_names
    result_df = pl.DataFrame(
        rows, schema=schema,
        orient="row"
    )

    agg_result = result_df \
        .group_by("r", "alpha") \
        .agg(
            [
                # We only aggregate value > -1 
                # (as -1 is assigned when the metric can't be calculated due to non definitive prediction)
                pl.col(col).filter(pl.col(col) > -1).mean()
                for col in col_names
            ]
        ) \
        .sort("r", "alpha")
    
    heights = {
        col: {
            "mean": [],
            "err_min": [],
            "err_max": []
        }
        for col in col_names
    }

    for alpha in alphas:
        arr_stats = agg_result \
            .filter(pl.col("alpha") == alpha) \
            .select(col_names) \
            .to_numpy()
        
        for i, col in enumerate(col_names):
            v = arr_stats[:, i].reshape(1, -1)
            nan_mask = np.isnan(v)
            v = v[~nan_mask].reshape(1, -1)
            bi = bootstrap(v, statistic=np.mean)
            ci = bi.confidence_interval

            min_ci = ci.low
            high_ci = ci.high
            mean_ci = bi.bootstrap_distribution.mean()

            # handling degenerate cases (no variation in prediction)
            if np.isnan(min_ci) | np.isnan(high_ci):
                min_ci = mean_ci
                high_ci = mean_ci

            heights[col]["mean"].append(high_ci)
            heights[col]["err_min"].append(mean_ci - min_ci)
            heights[col]["err_max"].append(high_ci - mean_ci)
    return heights