import polars as pl
from agg_utils import agg_case

def miscoverage_her2(df: pl.DataFrame, level: str) -> list:
    """
    A patch/case is not covered when
    - The final pred status is not equal to the label AND
    - The prediction size is less than 2 (as pred_size of 2 will always cover)
    """
    if level == "patch":
        pass
    elif level == "case":
        # aggregate prediction for case level
        df = agg_case(df)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'patch' or 'case'")
    
    miscover = df \
            .with_columns(
                pl.when(
                    (pl.col("final_pred") != pl.col("label")) &
                    (pl.col("pred_size") < 2)
                ).then(pl.lit(1))
                .otherwise(0)
                .alias("mis")
            ) \
            .group_by("label") \
            .agg(
                pl.col("mis").mean()
            ) \
            .sort("label").select("mis") \
            .to_numpy().ravel().tolist()
    return miscover

def miscoverage_group2_ihc(df: pl.DataFrame, level: str) -> list:
    """
    A patch/case is not covered when
    - The final pred status is not equal to the label AND
    - The prediction size is less than 2 (as pred_size of 2 will always cover)
    * Class: 2 (2+ DISH-) and 3 (2+ DISH+) is grouped to equivocal (2+)
    """
    if level == "patch":
        pass
    elif level == "case":
        # aggregate prediction for case level
        df = agg_case(df)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'patch' or 'case'")

    miscover = df \
            .with_columns(
                pl.when(
                    (pl.col("final_pred") != pl.col("label")) &
                    (pl.col("pred_size") < 2)
                ).then(pl.lit(1))
                .otherwise(0)
                .alias("mis")
            ) \
            .with_columns(
                pl.when(
                    pl.col("ihc_score") == 0
                ).then(pl.lit(0))
                .when(pl.col("ihc_score") == 1)
                .then(pl.lit(1))
                .when(pl.col("ihc_score").is_in([2, 3]))
                .then(pl.lit(2))
                .when(pl.col("ihc_score") == 4)
                .then(pl.lit(3))
                .alias("ihc_score")
            ) \
            .group_by("ihc_score") \
            .agg(
                pl.col("mis").mean()
            ) \
            .sort("ihc_score") \
            .select("mis") \
            .to_numpy().ravel().tolist()
    return miscover

def ambi_her2(df: pl.DataFrame, level: str) -> list:
    """
    A patch/case is classified as ambiguity when 
    the prediction set is not equal to 1.
    """
    if level == "patch":
        pass
    elif level == "case":
        # aggregate prediction for case level
        df = agg_case(df)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'patch' or 'case'")

    ambis = []
    for j in range(2):
        fdf = df \
            .filter(
                pl.col("label") == j
            )
        ambi = len(fdf.filter(pl.col("pred_size") != 1)) / len(df)
        ambis.append(ambi)
    return ambis

def ambi_group2_ihc(df: pl.DataFrame, level: str) -> list:
    """
    A patch/case is classified as ambiguity when 
    the prediction set is not equal to 1.
    * Class: 2 (2+ DISH-) and 3 (2+ DISH+) is grouped to equivocal (2+)    
    """
    if level == "patch":
        pass
    elif level == "case":
        # aggregate prediction for case level
        df = agg_case(df)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'patch' or 'case'")
    
    num_preds = []
    num_ambis = []
    ambis = []
    for j in range(5):
        fdf = df \
            .filter(
                pl.col("ihc_score") == j
            )
        
        num_pred = len(fdf)
        num_ambi = len(fdf.filter(pl.col("pred_size") != 1))

        num_preds.append(num_pred)
        num_ambis.append(num_ambi)
    
    # group 2+ patches
    num_preds = [
        num_preds[0],
        num_preds[1],
        num_preds[2] + num_preds[3],
        num_preds[4]
    ]
    num_ambis = [
        num_ambis[0],
        num_ambis[1],
        num_ambis[2] + num_ambis[3],
        num_ambis[4]
    ]

    for num_ambi, num_pred in list(zip(num_ambis, num_preds)):
        ambi = (num_ambi) / (num_pred + 1e-8)
        ambis.append(ambi)
    return ambis

def acc_her2(df: pl.DataFrame, level: str) -> list:
    """
    Accuracy of predicting correct HER2 status
    """
    if level == "patch":
        pass
    elif level == "case":
        # aggregate prediction for case level
        df = agg_case(df)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'patch' or 'case'")
    
    tp = len(df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 1)))
    tn = len(df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 0)))
    fp = len(df.filter((pl.col("final_pred") == 1) & (pl.col("label") == 0)))
    fn = len(df.filter((pl.col("final_pred") == 0) & (pl.col("label") == 1)))

    pos_acc = tp / (tp + fp + 1e-8) if tp + fp > 0 else - 1 
    # -1 will not be used for calculating acc (see Analysis/agg_utils.py -> agg_heights line 86)
    # Otherwise, the empty prediction will result in pos_acc = 0.
    # When using these values to calculate the avg. of acc, it will bias the value to 0.
    neg_acc = tn / (tn + fn + 1e-8) if tn + fn > 0 else - 1
    return [neg_acc, pos_acc]

def acc_group2_ihc(df: pl.DataFrame, level: str) -> list:
    """
    Accuracy of predicting correct HER2 status
    * Class: 2 (2+ DISH-) and 3 (2+ DISH+) is grouped to equivocal (2+)  
    """
    if level == "patch":
        pass
    elif level == "case":
        # aggregate prediction for case level
        df = agg_case(df)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'patch' or 'case'")
    
    # correct HER2 status for each IHC score (0-3)
    # key 2 represents 2+ (DISH-)
    # key 3 represents 2+ (DISH+)
    label_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1
    }

    num_preds = []
    corrects = []
    accs = []

    for score in [0, 1, 2, 3, 4]:
        sub_df = df \
            .filter(pl.col("ihc_score") == score) \
            .filter(pl.col("final_pred") != -1)
        num_pred = len(sub_df)
        correct = len(sub_df.filter(pl.col("final_pred") == label_dict[score]))

        num_preds.append(num_pred)
        corrects.append(correct)

    # group 2+ 
    num_preds = [
        num_preds[0],
        num_preds[1],
        num_preds[2] + num_preds[3],
        num_preds[4]
    ]

    corrects = [
        corrects[0],
        corrects[1],
        corrects[2] + corrects[3],
        corrects[4]
    ]

    for correct, num_pred in list(zip(corrects, num_preds)):
        acc = correct / (num_pred + 1e-8) if num_pred > 0 else -1
        accs.append(acc)
    
    return accs

def acc_equivocal_ihc(df: pl.DataFrame, level: str) -> list:
    """
    Accuracy of predicting correct HER2 status
    * For 2+ only (No group)  
    """
    if level == "patch":
        pass
    elif level == "case":
        # aggregate prediction for case level
        df = agg_case(df)
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'patch' or 'case'")

    # correct HER2 status for each IHC score (0-3)
    # key 2 represents 2+ (DISH-)
    # key 3 represents 2+ (DISH+)
    label_dict = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1
    }

    accs = []
    for score in [2, 3]:
        sub_df = df \
            .filter(pl.col("ihc_score") == score) \
            .filter(pl.col("final_pred") != -1)
        num_pred = len(sub_df)
        correct = len(sub_df.filter(pl.col("final_pred") == label_dict[score]))
        acc = correct / (num_pred + 1e-8) if num_pred > 0 else -1
        accs.append(acc)
    return accs

        


            