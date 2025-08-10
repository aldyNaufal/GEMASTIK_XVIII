import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import numpy as np

# 1. Memuat Dataset
try:
    df = pd.read_csv('./data/main_data.csv')

    # 2. Pra-pemrosesan Data
    df_cleaned = df.drop(columns=['Luas Panen Padi', 'Potensi Gagal Panen', 'year_month', 'Datetime'])

    # 3. Mendefinisikan Fitur dan Target
    X = df_cleaned.drop(columns=['Produksi Padi', 'Beras'])
    y = df_cleaned[['Produksi Padi', 'Beras']]

    # 4. Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Membagi Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- Hyperparameter Tuning untuk XGBoost ---
    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_grid,
        n_iter=20,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    xgb_model = MultiOutputRegressor(xgb_search)
    xgb_model.fit(X_train, y_train)

    # --- Hyperparameter Tuning untuk Random Forest ---
    rf_base = RandomForestRegressor(random_state=42)

    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_grid,
        n_iter=20,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    rf_model = MultiOutputRegressor(rf_search)
    rf_model.fit(X_train, y_train)

    # Simpan model & scaler
    with open('xgboost_tuned.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open('random_forest_tuned.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # --- Evaluasi ---
    def evaluate_model(name, model):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        return mse, r2

    mse_xgb, r2_xgb = evaluate_model("XGBoost Tuned", xgb_model)
    mse_rf, r2_rf = evaluate_model("Random Forest Tuned", rf_model)

    # --- Feature Importance untuk XGBoost ---
    # Ambil best estimator dari search untuk masing-masing target
    importances_padi = xgb_model.estimators_[0].best_estimator_.feature_importances_
    importances_beras = xgb_model.estimators_[1].best_estimator_.feature_importances_
    avg_importances = (importances_padi + importances_beras) / 2

    feature_names = X.columns
    importance_df = pd.DataFrame({'Fitur': feature_names, 'Tingkat Kepentingan': avg_importances})
    importance_df = importance_df.sort_values(by='Tingkat Kepentingan', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Fitur'], importance_df['Tingkat Kepentingan'])
    plt.xlabel('Tingkat Kepentingan')
    plt.ylabel('Fitur')
    plt.title('Tingkat Kepentingan Fitur (XGBoost Tuned)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('xgboost_tuned_feature_importance.png')

    print("\n✅ Tuning & scaling selesai. Model disimpan.")
    print("✅ Plot tingkat kepentingan fitur disimpan sebagai 'xgboost_tuned_feature_importance.png'")

except FileNotFoundError:
    print("File 'main_data.csv' tidak ditemukan.")
except ImportError:
    print("Error: Pastikan library yang dibutuhkan sudah diinstall.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
