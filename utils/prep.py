import polars as pl

def prep_case(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("case").str.contains("1+", literal=True))
        .then(pl.lit("1+"))
        .when(pl.col("case").str.contains("score 0 case 2", literal=True))
        .then(pl.lit("0"))
        .when(pl.col("case").str.contains("3+ D+ 01", literal=True))
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("2+ DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("3+", literal=True))
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("28 Jun HER2 IHC negative", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ D+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ DISH -", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("HER2 0", literal=True))
        .then(pl.lit("0"))
        .when(pl.col("case").str.contains("HER2 score 1", literal=True))
        .then(pl.lit("1+"))
        .when(pl.col("case").str.contains("2+ DISH-", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ Dish -", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2+ DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ DISH +", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("2+ Dish+", literal=True))
        .then(pl.lit("2+"))
        .when(
            pl.col("case").str.contains(
                "13 Sep HER2 different brightness", literal=True
            )
        )
        .then(pl.lit("3+"))
        .when(pl.col("case").str.contains("2+DISH+", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("HER2 neg case 01", literal=True))
        .then(pl.lit("2-"))
        .when(pl.col("case").str.contains("2 + DISH +", literal=True))
        .then(pl.lit("2+"))
        .when(pl.col("case").str.contains("score 0", literal=True))
        .then(pl.lit("0"))
        .otherwise(None)
        .alias("ihc_score")
    ).with_columns(
        pl.when(pl.col("ihc_score").is_in(["0", "1+", "2-"]))
        .then(pl.lit(0))
        .otherwise(1)
        .alias("label")
    )
    return df