"""
Dataset normalization script for NASA C-MAPSS (FD001).

Labels:
    0 -> Safe      (RUL > 100)
    1 -> Warning   (30 < RUL <= 100)
    2 -> Critical  (RUL <= 30)
"""

import os
import pandas as pd


def assign_label(rul):
    """Map a RUL value to a discrete health-state label."""
    if rul <= 30:
        return 2      # Critical
    elif rul <= 100:
        return 1      # Warning
    else:
        return 0      # Safe


def build_dataset(data_dir="."):
    train_path = os.path.join(data_dir, "train_FD001.txt")
    test_path = os.path.join(data_dir, "test_FD001.txt")
    rul_path = os.path.join(data_dir, "RUL_FD001.txt")

    # ---- Load raw train set -------------------------------------------------
    df_train = pd.read_csv(train_path, sep=r"\s+", header=None)
    df_train["kind"] = "train"
    df_train.rename(columns={0: "sample", 1: "cycles"}, inplace=True)
    df_train["RUL"] = (
        df_train.groupby("sample")["cycles"].transform("max") - df_train["cycles"]
    )

    # ---- Load raw test set ---------------------------------------------------
    df_test = pd.read_csv(test_path, sep=r"\s+", header=None)
    df_test["kind"] = "test"
    df_test.rename(columns={0: "sample", 1: "cycles"}, inplace=True)
    df_test["RUL"] = (
        df_test.groupby("sample")["cycles"].transform("max") - df_test["cycles"]
    )

    # ---- True RUL at the last recorded cycle of every test engine -----------
    df_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)

    for index, row in df_test.iterrows():
        rul_idx = row["sample"] - 1
        df_test.at[index, "RUL"] += df_rul.iloc[rul_idx, 0]
        if row["sample"] <= 50:
            df_test.at[index, "kind"] = "validation"

    df = pd.concat([df_train, df_test], axis=0)
    df.reset_index(drop=True, inplace=True)

    # ---- Rename raw sensor columns -------------------------------------------
    sensors = [f"sensor{i}" for i in range(1, 25)]
    df.rename(columns={i: sensors[i - 2] for i in range(2, 26)}, inplace=True)

    # ---- Remove sensors that are constant on the training split --------------
    train_data = df[df["kind"] == "train"]
    train_min = train_data[sensors].min()
    train_max = train_data[sensors].max()

    constant_sensors = train_min[train_min == train_max].index.tolist()
    print(f"Constant sensors removed: {constant_sensors}")

    with open(os.path.join(data_dir, "removed_constant_sensors.txt"), "w") as f:
        f.write("\n".join(constant_sensors))

    df.drop(columns=constant_sensors, inplace=True)
    active_sensors = [s for s in sensors if s not in constant_sensors]

    # ---- Min-Max normalization (parameters fit on train split only) ---------
    df[active_sensors] = (df[active_sensors] - train_min[active_sensors]) / (
        train_max[active_sensors] - train_min[active_sensors]
    )

    # ---- Health-state label ---------------------------------------------------
    df["label"] = df["RUL"].apply(assign_label)

    out_path = os.path.join(data_dir, "dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"dataset.csv written to {out_path} ({len(df)} rows, "
          f"{len(active_sensors)} active sensors)")
    return df


if __name__ == "__main__":
    build_dataset(data_dir=os.path.dirname(os.path.abspath(__file__)))
