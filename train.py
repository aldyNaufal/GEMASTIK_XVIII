import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb  # Mengimpor library XGBoost
import pickle
import matplotlib.pyplot as plt

# 1. Memuat Dataset
try:
    df = pd.read_csv('./data/main_data.csv')

    # 2. Pra-pemrosesan Data
    df_cleaned = df.drop(columns=['Luas Panen Padi', 'Potensi Gagal Panen', 'year_month', 'Datetime'])

    # 3. Mendefinisikan Fitur (X) dan Target (y)
    X = df_cleaned.drop(columns=['Produksi Padi', 'Beras'])
    y = df_cleaned[['Produksi Padi', 'Beras']]

    # 4. Membagi Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Membuat dan Melatih Model XGBoost
    # Inisiasi model XGBRegressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    
    # Bungkus dengan MultiOutputRegressor untuk menangani multi-target
    model = MultiOutputRegressor(estimator=xgb_regressor)
    
    # Latih model dengan data training
    model.fit(X_train, y_train)

    # 6. Menyimpan Model ke File Pickle
    model_filename = 'xgboost_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"✅ Model telah berhasil disimpan ke file: {model_filename}")

    # 7. Melakukan Prediksi pada Data Uji
    y_pred = model.predict(X_test)

    # 8. Mengevaluasi Kinerja Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Evaluasi Model XGBoost ---")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # 9. Menampilkan Tingkat Kepentingan Fitur (Rata-rata dari kedua model)
    # Ambil tingkat kepentingan dari masing-masing model target
    importances_padi = model.estimators_[0].feature_importances_
    importances_beras = model.estimators_[1].feature_importances_
    
    # Hitung rata-rata tingkat kepentingan
    avg_importances = (importances_padi + importances_beras) / 2

    feature_names = X.columns
    importance_df = pd.DataFrame({'Fitur': feature_names, 'Tingkat Kepentingan': avg_importances})
    importance_df = importance_df.sort_values(by='Tingkat Kepentingan', ascending=False)

    print("\n--- Rata-rata Tingkat Kepentingan Fitur ---")
    print(importance_df)
    
    # Membuat dan menyimpan plot tingkat kepentingan fitur
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Fitur'], importance_df['Tingkat Kepentingan'])
    plt.xlabel('Tingkat Kepentingan')
    plt.ylabel('Fitur')
    plt.title('Tingkat Kepentingan Fitur pada Model XGBoost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    
    print("\n✅ Plot tingkat kepentingan fitur disimpan sebagai 'xgboost_feature_importance.png'")

except FileNotFoundError:
    print("File 'main_data.csv' tidak ditemukan.")
except ImportError:
    print("Error: Library xgboost tidak terinstal. Silakan instal dengan 'pip install xgboost'")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")