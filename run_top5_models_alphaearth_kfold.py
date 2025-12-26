#!/usr/bin/env python3
"""
Run Top 5 Rainy Season Models with AlphaEarth - K-Fold Cross-Validation
For fair comparison with baseline results
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import distance_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, MultiHeadAttention, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
import rasterio
from rasterio.windows import Window
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/Users/rakibhhridoy/Five_Rivers")
GIS_ROOT = PROJECT_ROOT / "gis" / "SedimentRainyAE"
DATA_ROOT = PROJECT_ROOT / "data"

# AlphaEarth data files
RAINY_DATA_FILE = GIS_ROOT / "Option_B_RainyAE.csv"
RASTER_CAL_INDICES = GIS_ROOT.parent / "CalIndices" / "*.tif"
RASTER_LULC = GIS_ROOT.parent / "LULCMerged" / "*.tif"
RASTER_IDW = GIS_ROOT.parent / "IDW" / "*.tif"

BUFFER_METERS = 500
BATCH_SIZE = 4
N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load AlphaEarth data with all features"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Load AlphaEarth combined data
    print(f"Loading: {RAINY_DATA_FILE}")
    data = pd.read_csv(RAINY_DATA_FILE)
    print(f"  ✓ Loaded {len(data)} samples with {len(data.columns)} features")

    # Separate features and target
    drop_cols = ['Stations', 'River', 'Lat', 'Long', 'geometry']
    feature_cols = [c for c in data.columns if c not in drop_cols + ['RI']]

    X = data[feature_cols].values
    y = data['RI'].values
    coords = data[['Long', 'Lat']].values

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(y)}")
    print(f"  Target range: {y.min():.2f} - {y.max():.2f}")

    return X, y, coords, feature_cols

def load_rasters():
    """Load raster files for CNN input"""
    print("\nLoading raster layers...")
    raster_paths = []
    raster_paths += glob.glob(str(RASTER_CAL_INDICES))
    raster_paths += glob.glob(str(RASTER_LULC))
    raster_paths += glob.glob(str(RASTER_IDW))

    print(f"  ✓ Found {len(raster_paths)} raster layers")
    return sorted(raster_paths)

class RasterDataGenerator:
    """Generate CNN patches from rasters"""
    def __init__(self, raster_files, sample_coords, patch_size=100):
        self.raster_files = raster_files
        self.coords = sample_coords
        self.patch_size = patch_size

    def get_patches(self, sample_indices):
        """Extract CNN patches for given sample indices"""
        batch_patches = []
        for idx in sample_indices:
            lon, lat = self.coords[idx]
            channels = []

            for rfile in self.raster_files:
                try:
                    with rasterio.open(rfile) as src:
                        row, col = src.index(lon, lat)
                        left = max(0, int(col - self.patch_size/2))
                        top = max(0, int(row - self.patch_size/2))
                        right = min(src.width, int(col + self.patch_size/2))
                        bottom = min(src.height, int(row + self.patch_size/2))

                        window = Window(left, top, right-left, bottom-top)
                        patch = src.read(1, window=window)

                        if patch.shape != (self.patch_size, self.patch_size):
                            padded = np.zeros((self.patch_size, self.patch_size))
                            padded[:patch.shape[0], :patch.shape[1]] = patch
                            patch = padded

                        channels.append(patch)
                except Exception as e:
                    channels.append(np.zeros((self.patch_size, self.patch_size)))

            if channels:
                batch_patches.append(np.stack(channels, axis=-1))

        return np.array(batch_patches) if batch_patches else np.zeros((len(sample_indices), self.patch_size, self.patch_size, 1))

# ============================================================================
# MODELS
# ============================================================================

def build_transformer_model(cnn_shape, gnn_dim, mlp_dim):
    """Transformer CNN GNN MLP Fusion Model"""
    cnn_input = Input(shape=cnn_shape, name='cnn_input')
    x_cnn = Conv2D(16, 3, activation='relu', padding='same')(cnn_input)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Conv2D(32, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Conv2D(64, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(128, activation='relu')(x_cnn)
    x_cnn = Dropout(0.3)(x_cnn)

    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')
    x_mlp = Dense(64, activation='relu')(mlp_input)
    x_mlp = Dropout(0.3)(x_mlp)
    x_mlp = Dense(32, activation='relu')(x_mlp)

    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')
    x_gnn = Dense(32, activation='relu')(gnn_input)
    x_gnn = Dropout(0.3)(x_gnn)

    x_cnn_attn = Reshape((16, 8))(x_cnn)
    x_mlp_attn = Reshape((32, 1))(x_mlp)
    x_gnn_attn = Reshape((32, 1))(x_gnn)

    attn_output = MultiHeadAttention(num_heads=4, key_dim=8)(
        x_cnn_attn, Concatenate()([x_mlp_attn, x_gnn_attn])
    )
    x_fused = Flatten()(attn_output)
    x_fused = Dense(128, activation='relu')(x_fused)
    x_fused = Dropout(0.3)(x_fused)

    output = Dense(1)(x_fused)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_gnn_mlp_ae_model(mlp_dim, gnn_dim):
    """GNN MLP AutoEncoder"""
    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')
    x_mlp = Dense(64, activation='relu')(mlp_input)
    x_mlp = Dense(32, activation='relu')(x_mlp)

    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')
    x_gnn = Dense(32, activation='relu')(gnn_input)
    x_gnn = Dense(16, activation='relu')(x_gnn)

    x = Concatenate()([x_mlp, x_gnn])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_cnn_gnn_mlp_pg_model(cnn_shape, gnn_dim, mlp_dim):
    """CNN GNN MLP Progressive"""
    cnn_input = Input(shape=cnn_shape, name='cnn_input')
    x_cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Conv2D(64, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(128, activation='relu')(x_cnn)

    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')
    x_mlp = Dense(64, activation='relu')(mlp_input)

    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')
    x_gnn = Dense(32, activation='relu')(gnn_input)

    x = Concatenate()([x_cnn, x_mlp, x_gnn])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1)(x)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_gnn_mlp_model(mlp_dim, gnn_dim):
    """Simple GNN MLP"""
    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')
    x_mlp = Dense(64, activation='relu')(mlp_input)
    x_mlp = Dropout(0.2)(x_mlp)
    x_mlp = Dense(32, activation='relu')(x_mlp)

    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')
    x_gnn = Dense(32, activation='relu')(gnn_input)
    x_gnn = Dropout(0.2)(x_gnn)

    x = Concatenate()([x_mlp, x_gnn])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1)(x)

    model = Model(inputs=[mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_stacked_cnn_gnn_mlp_model(cnn_shape, gnn_dim, mlp_dim):
    """Stacked CNN GNN MLP"""
    cnn_input = Input(shape=cnn_shape, name='cnn_input')
    x_cnn = Conv2D(32, 3, activation='relu', padding='same')(cnn_input)
    x_cnn = Conv2D(64, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Conv2D(128, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Flatten()(x_cnn)
    x_cnn = Dense(256, activation='relu')(x_cnn)
    x_cnn = Dropout(0.4)(x_cnn)
    x_cnn = Dense(128, activation='relu')(x_cnn)

    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')
    x_mlp = Dense(128, activation='relu')(mlp_input)
    x_mlp = Dropout(0.3)(x_mlp)
    x_mlp = Dense(64, activation='relu')(x_mlp)
    x_mlp = Dropout(0.3)(x_mlp)
    x_mlp = Dense(32, activation='relu')(x_mlp)

    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')
    x_gnn = Dense(64, activation='relu')(gnn_input)
    x_gnn = Dropout(0.3)(x_gnn)
    x_gnn = Dense(32, activation='relu')(x_gnn)

    x = Concatenate()([x_cnn, x_mlp, x_gnn])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(1)(x)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    smape = np.mean(diff[denominator != 0]) * 100 if np.any(denominator != 0) else 0

    return r2, rmse, mae, smape

def evaluate_model(model, mlp_data, gnn_data, y_true, raster_gen=None, test_indices=None, has_cnn=True):
    """Evaluate model and get predictions"""
    if has_cnn:
        cnn_patches = raster_gen.get_patches(test_indices)
        y_pred = model.predict([cnn_patches, mlp_data, gnn_data], verbose=0)
    else:
        y_pred = model.predict([mlp_data, gnn_data], verbose=0)

    return y_pred.flatten()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("RUNNING TOP 5 RAINY SEASON MODELS WITH ALPHAEARTH")
    print("K-Fold Cross-Validation (5 Folds)")
    print("="*80)

    # Load data
    X, y, coords, feature_cols = load_data()
    raster_paths = load_rasters()

    # K-Fold setup
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION SETUP")
    print("="*80)
    print(f"Total samples: {len(y)}")
    print(f"Number of folds: {N_SPLITS}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Model configurations
    models_config = [
        {
            "name": "1. Transformer CNN GNN MLP",
            "has_cnn": True
        },
        {
            "name": "2. GNN MLP AE",
            "has_cnn": False
        },
        {
            "name": "3. CNN GNN MLP PG",
            "has_cnn": True
        },
        {
            "name": "4. GNN MLP",
            "has_cnn": False
        },
        {
            "name": "5. Stacked CNN GNN MLP",
            "has_cnn": True
        },
    ]

    # Results storage
    all_results = {model["name"]: {"r2_scores": [], "rmse_scores": [], "mae_scores": [], "smape_scores": []}
                   for model in models_config}

    # K-Fold loop
    fold_num = 0
    for train_idx, test_idx in kf.split(X):
        fold_num += 1
        print(f"\n{'='*80}")
        print(f"FOLD {fold_num}")
        print(f"{'='*80}")
        print(f"Training samples: {len(train_idx)}")
        print(f"Test samples: {len(test_idx)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Prepare GNN features
        dist_matrix_train = distance_matrix(X_train_scaled, X_train_scaled)
        dist_matrix_test = distance_matrix(X_test_scaled, X_train_scaled)
        gnn_train_data = dist_matrix_train.mean(axis=1, keepdims=True)
        gnn_test_data = dist_matrix_test.mean(axis=1, keepdims=True)

        # Raster generator
        raster_gen = RasterDataGenerator(raster_paths, coords, patch_size=100)
        cnn_shape = (100, 100, len(raster_paths))

        # Run each model
        for config in models_config:
            if config["name"] == "1. Transformer CNN GNN MLP":
                model = build_transformer_model(cnn_shape, gnn_test_data.shape[1], X_train_scaled.shape[1])
            elif config["name"] == "2. GNN MLP AE":
                model = build_gnn_mlp_ae_model(X_train_scaled.shape[1], gnn_test_data.shape[1])
            elif config["name"] == "3. CNN GNN MLP PG":
                model = build_cnn_gnn_mlp_pg_model(cnn_shape, gnn_test_data.shape[1], X_train_scaled.shape[1])
            elif config["name"] == "4. GNN MLP":
                model = build_gnn_mlp_model(X_train_scaled.shape[1], gnn_test_data.shape[1])
            else:  # Stacked CNN GNN MLP
                model = build_stacked_cnn_gnn_mlp_model(cnn_shape, gnn_test_data.shape[1], X_train_scaled.shape[1])

            # Train
            if config["has_cnn"]:
                cnn_patches = raster_gen.get_patches(train_idx)
                model.fit(
                    [cnn_patches, X_train_scaled, gnn_train_data],
                    y_train,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)]
                )
            else:
                model.fit(
                    [X_train_scaled, gnn_train_data],
                    y_train,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)]
                )

            # Evaluate
            y_pred = evaluate_model(model, X_test_scaled, gnn_test_data, y_test, raster_gen, test_idx, config["has_cnn"])
            r2, rmse, mae, smape = calculate_metrics(y_test, y_pred)

            all_results[config["name"]]["r2_scores"].append(r2)
            all_results[config["name"]]["rmse_scores"].append(rmse)
            all_results[config["name"]]["mae_scores"].append(mae)
            all_results[config["name"]]["smape_scores"].append(smape)

            print(f"  {config['name']}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL MODELS (K-FOLD AVERAGED)")
    print("="*80)

    results = []
    for model_name in [m["name"] for m in models_config]:
        avg_r2 = np.mean(all_results[model_name]["r2_scores"])
        avg_rmse = np.mean(all_results[model_name]["rmse_scores"])
        avg_mae = np.mean(all_results[model_name]["mae_scores"])
        avg_smape = np.mean(all_results[model_name]["smape_scores"])

        print(f"\n{model_name}")
        print(f"  R² Scores across folds: {[f'{s:.4f}' for s in all_results[model_name]['r2_scores']]}")
        print(f"  Average R²:    {avg_r2:.4f}")
        print(f"  Average RMSE:  {avg_rmse:.4f}")
        print(f"  Average MAE:   {avg_mae:.4f}")
        print(f"  Average SMAPE: {avg_smape:.2f}%")

        results.append({
            'Model': model_name,
            'R² (avg)': avg_r2,
            'R² (std)': np.std(all_results[model_name]["r2_scores"]),
            'RMSE (avg)': avg_rmse,
            'MAE (avg)': avg_mae,
            'SMAPE (avg)': avg_smape
        })

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # Save results
    output_file = PROJECT_ROOT / "TOP5_RAINY_ALPHAEARTH_KFOLD_RESULTS.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
