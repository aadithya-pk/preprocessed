import time
import os
import pandas as pd
import lightgbm as lgb
from data_utils import load_tabular_data
from evaluate_metrics import compute_metrics, save_predictions, RESULTS_DIR

def train_and_evaluate_lgbm():
    print("Loading preprocessed dataset splits...")
    data = load_tabular_data()

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    feature_names = data['feature_names']
    test_index = data['test_df'].index

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    print("\n--- Training Model 1: LightGBM (Enhanced Tabular GBDT) ---")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2024,
        n_jobs=-1,
        importance_type='gain'
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    elapsed_time = time.time() - t0

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics & update leaderboard
    compute_metrics(y_test, y_pred, model_name="LightGBM", training_time_sec=elapsed_time)

    # Save predictions
    save_predictions(y_test, y_pred, test_index, model_name="LightGBM")

    # Save feature importances
    df_feat = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': model.feature_importances_
    }).sort_values('importance_gain', ascending=False)
    feat_path = os.path.join(RESULTS_DIR, "lightgbm_feature_importance.csv")
    df_feat.to_csv(feat_path, index=False)
    print(f"Feature importance saved to: {feat_path}")
    print("\nTop 5 Features according to LightGBM:")
    print(df_feat.head(5).to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate_lgbm()
