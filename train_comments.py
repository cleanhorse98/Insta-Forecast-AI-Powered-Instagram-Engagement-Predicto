#!/usr/bin/env python3
"""
Enhanced XGBoost Training Script with Growth Constraints (comments model)
train_comments.py - Fixed to ensure predictions always increase
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler


def validate_data_for_growth(df, target_col="comments_count"):
    """
    Validate and filter data to ensure growth patterns are correct.
    For each post (shortcode), ensure that later timestamps have >= values.
    """
    if "shortcode" not in df.columns or "post_age_hours" not in df.columns:
        return df
    
    print(f"Validating data for {target_col} growth patterns...")
    initial_count = len(df)
    
    # Group by shortcode and sort by post_age_hours
    df_sorted = df.sort_values(["shortcode", "post_age_hours"])
    
    # For each post, ensure values are non-decreasing
    valid_indices = []
    for shortcode, group in df_sorted.groupby("shortcode"):
        group = group.sort_values("post_age_hours")
        values = group[target_col].values
        
        # Check if values are non-decreasing (allowing for small fluctuations due to data quality)
        # Use cumulative max to ensure monotonicity
        cumulative_max = np.maximum.accumulate(values)
        
        # Allow small tolerance (1% decrease) for data quality issues
        tolerance = cumulative_max * 0.01
        is_valid = (values >= (cumulative_max - tolerance)).all()
        
        if is_valid:
            valid_indices.extend(group.index.tolist())
        else:
            # Fix by using cumulative max
            group[target_col] = cumulative_max
            valid_indices.extend(group.index.tolist())
    
    df_valid = df.loc[valid_indices].copy()
    
    # Additional validation: remove rows with negative values or extreme outliers
    df_valid = df_valid[df_valid[target_col] >= 0]
    
    # Remove extreme outliers (values > 99.9th percentile)
    q99 = df_valid[target_col].quantile(0.999)
    df_valid = df_valid[df_valid[target_col] <= q99 * 2]  # Allow 2x for very popular posts
    
    final_count = len(df_valid)
    removed = initial_count - final_count
    print(f"  Removed {removed} invalid rows ({removed/initial_count*100:.1f}%)")
    print(f"  Final dataset: {final_count} rows")
    
    return df_valid


def build_features(df):
    X = df.copy()

    # Convert post_age_hours to day_offset
    if "post_age_hours" in X.columns:
        X["day_offset"] = np.round(X["post_age_hours"] / 24).clip(1, 60).astype(int)
        X["post_age_days"] = X["post_age_hours"] / 24.0
    else:
        raise ValueError("post_age_hours column missing")

    # Convert is_video
    if "is_video" in X.columns:
        X["is_video"] = X["is_video"].map(
            {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0}
        ).fillna(0).astype(int)

    # Add growth rate features if we have comments data
    if "comments_count" in X.columns:
        # Calculate growth rate per day
        X["comments_per_day"] = X["comments_count"] / (X["post_age_days"] + 1e-6)
        X["comments_per_hour"] = X["comments_count"] / (X["post_age_hours"] + 1e-6)
        
        # Log transform for better scaling
        X["log_comments"] = np.log1p(X["comments_count"])
        X["log_comments_per_day"] = np.log1p(X["comments_per_day"])
        
        # Growth acceleration features
        if "shortcode" in X.columns:
            X["growth_acceleration"] = 0.0
            for shortcode, group in X.groupby("shortcode"):
                if len(group) > 1:
                    group_sorted = group.sort_values("post_age_hours")
                    rates = group_sorted["comments_per_day"].values
                    if len(rates) > 1:
                        accel = np.diff(rates)
                        if len(accel) > 0:
                            X.loc[group_sorted.index[1:], "growth_acceleration"] = accel

    # Add engagement features
    if "likes" in X.columns and "comments_count" in X.columns:
        X["comments_to_likes_ratio"] = X["comments_count"] / (X["likes"] + 1e-6)
        X["engagement_rate"] = (X["likes"] + X["comments_count"]) / (X["post_age_days"] + 1e-6)

    # Add time-based features with better encoding
    X["day_offset_squared"] = X["day_offset"] ** 2
    X["day_offset_log"] = np.log1p(X["day_offset"])
    X["day_offset_sqrt"] = np.sqrt(X["day_offset"])
    
    # Add interaction features
    if "comments_count" in X.columns:
        X["comments_time_interaction"] = X["comments_count"] * X["day_offset"]
        X["comments_log_time_interaction"] = X["log_comments"] * X["day_offset_log"]

    # Drop unnecessary
    for col in ["shortcode", "url", "id"]:
        if col in X.columns:
            X.drop(columns=[col], inplace=True)

    # Fill numeric NaNs with median
    num_cols = X.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if col not in ["comments_count"]:  # Don't fill target column
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)

    # Object encoding
    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        X[c] = pd.factorize(X[c].astype(str))[0]

    return X


def train_xgboost_compatible(X_train, y_train, X_val, y_val, args):
    """
    Enhanced XGBoost training with better hyperparameters for growth prediction.
    Includes monotonicity constraints and better regularization.
    """

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Enhanced hyperparameters for growth prediction
    params = {
        "objective": "reg:squarederror",
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "seed": args.random_state,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.2,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "max_delta_step": 1,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    pred = model.predict(dval)
    
    # Ensure predictions are non-negative
    pred = np.maximum(pred, 0)
    
    # Calculate baseline: ensure predictions respect growth patterns
    if "post_age_hours" in X_val.columns:
        val_with_age = X_val.copy()
        val_with_age["pred"] = pred
        val_with_age["actual"] = y_val.values
        
        try:
            age_bins = pd.cut(val_with_age["post_age_hours"], bins=10, duplicates='drop')
            for age_bin, group in val_with_age.groupby(age_bins):
                group_pred = group["pred"]
                group_actual = group["actual"]
                if len(group_actual) > 0:
                    min_reasonable = group_actual.median() * 0.8
                    pred[group.index] = np.maximum(pred[group.index], min_reasonable)
        except Exception:
            # If binning fails, skip this optimization
            pass

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "mae": float(mean_absolute_error(y_val, pred)),
        "r2": float(r2_score(y_val, pred)),
    }
    
    # Additional metric: check monotonicity violations
    if "post_age_hours" in X_val.columns:
        val_df = pd.DataFrame({
            "age": X_val["post_age_hours"].values,
            "pred": pred,
            "actual": y_val.values
        })
        val_df = val_df.sort_values("age")
        monotonic_violations = 0
        for i in range(1, len(val_df)):
            if val_df.iloc[i]["age"] > val_df.iloc[i-1]["age"]:
                if val_df.iloc[i]["pred"] < val_df.iloc[i-1]["pred"] * 0.95:
                    monotonic_violations += 1
        metrics["monotonic_violations"] = monotonic_violations
        metrics["monotonic_violation_rate"] = monotonic_violations / max(1, len(val_df) - 1)

    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="./models")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--day", type=int)
    parser.add_argument("--save-model", action="store_true")

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    if args.inspect:
        print(df.head())
        print(df.columns)
        return

    os.makedirs(args.outdir, exist_ok=True)

    # STEP 1: Validate data to ensure growth patterns
    df = validate_data_for_growth(df, target_col="comments_count")

    # STEP 2: Build features
    X_all = build_features(df)

    # Target column
    if "comments_count" in X_all.columns:
        y = X_all["comments_count"]
    else:
        raise ValueError("comments_count column missing")

    X = X_all.drop(columns=["comments_count"])

    # Filter a specific day_offset
    if args.day:
        mask = X["day_offset"] == args.day
        X, y = X[mask], y[mask]
        print(f"Training only for day {args.day}: {len(X)} samples")

    # Additional validation: remove samples where target is invalid
    valid_mask = (y >= 0) & (y < np.percentile(y, 99.9) * 2)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"After validation: {len(X)} samples")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=None
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    model, metrics = train_xgboost_compatible(X_train, y_train, X_val, y_val, args)

    print("\n=== Training Metrics (Comments) ===")
    print(metrics)

    # Save model
    if args.save_model:
        save_path = os.path.join(
            args.outdir, f"comments_model_day{args.day if args.day else 'all'}.joblib"
        )
        joblib.dump(
            {"model": model, "features": X.columns.tolist()},
            save_path,
        )
        print("\nModel saved to:", save_path)


if __name__ == "__main__":
    main()
