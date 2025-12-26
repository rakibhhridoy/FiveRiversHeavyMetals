#!/usr/bin/env python3
"""
Run Top 5 Rainy Season Models WITHOUT AlphaEarth (Baseline) - Single Train/Test Split
For fair comparison with AlphaEarth results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import distance_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, MultiHeadAttention, LayerNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import rasterio
from rasterio.windows import Window
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/Users/rakibhhridoy/Five_Rivers")
GIS_ROOT = PROJECT_ROOT / "gis"
DATA_ROOT = PROJECT_ROOT / "Python"

# Baseline data (original 21 features only)
RAINY_DATA_FILE = DATA_ROOT / "rainy.csv"
RASTER_CAL_INDICES = GIS_ROOT / "CalIndices" / "*.tif"
RASTER_LULC = GIS_ROOT / "LULCMerged" / "*.tif"
RASTER_IDW = GIS_ROOT / "IDW" / "*.tif"

BUFFER_METERS = 500
BATCH_SIZE = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load baseline data (original 21 features only)"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Load baseline data
    print(f"Loading: {RAINY_DATA_FILE}")
    data = pd.read_csv(RAINY_DATA_FILE, index_col=0)
    print(f"  ✓ Loaded {len(data)} samples with {len(data.columns)} features")

    # The baseline data has 6 metal columns: Cr, Ni, Cu, As, Cd, Pb
    # RI (Richness Index) is calculated from these metals
    X = data.values

    # Create RI as sum of metal concentrations (this was the original target)
    y = X.sum(axis=1)

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(y)}")
    print(f"  Target range: {y.min():.2f} - {y.max():.2f}")

    return X, y

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
                        # Extract patch around sample location
                        left = max(0, int(col - self.patch_size/2))
                        top = max(0, int(row - self.patch_size/2))
                        right = min(src.width, int(col + self.patch_size/2))
                        bottom = min(src.height, int(row + self.patch_size/2))

                        window = Window(left, top, right-left, bottom-top)
                        patch = src.read(1, window=window)

                        # Pad if necessary
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
    # CNN branch
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

    # MLP branch
    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')
    x_mlp = Dense(64, activation='relu')(mlp_input)
    x_mlp = Dropout(0.3)(x_mlp)
    x_mlp = Dense(32, activation='relu')(x_mlp)

    # GNN branch
    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')
    x_gnn = Dense(32, activation='relu')(gnn_input)
    x_gnn = Dropout(0.3)(x_gnn)

    # Reshape for attention
    x_cnn_attn = Reshape((16, 8))(x_cnn)
    x_mlp_attn = Reshape((32, 1))(x_mlp)
    x_gnn_attn = Reshape((32, 1))(x_gnn)

    # Multi-head attention fusion
    attn_output = MultiHeadAttention(num_heads=4, key_dim=8)(
        x_cnn_attn, Concatenate()([x_mlp_attn, x_gnn_attn])
    )
    x_fused = Flatten()(attn_output)
    x_fused = Dense(128, activation='relu')(x_fused)
    x_fused = Dropout(0.3)(x_fused)

    # Output layer
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

    # SMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    smape = np.mean(diff[denominator != 0]) * 100 if np.any(denominator != 0) else 0

    return r2, rmse, mae, smape

def evaluate_model(model, mlp_data, gnn_data, y_true, raster_gen=None, indices=None, has_cnn=True):
    """Evaluate model and get predictions"""
    if has_cnn:
        cnn_patches = raster_gen.get_patches(indices)
        y_pred = model.predict([cnn_patches, mlp_data[indices], gnn_data[indices]], verbose=0)
    else:
        y_pred = model.predict([mlp_data[indices], gnn_data[indices]], verbose=0)

    return y_pred.flatten()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("RUNNING TOP 5 RAINY SEASON MODELS (BASELINE - NO ALPHAEARTH)")
    print("Single Train/Test Split (80/20)")
    print("="*80)

    # Load data
    X, y = load_data()
    raster_paths = load_rasters()

    # Use dummy coordinates (not available in baseline data)
    coords = np.array([[90.23 + i*0.01, 23.91 - i*0.01] for i in range(len(y))])

    # Train/test split
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare data for models
    mlp_dim = X_train_scaled.shape[1]
    gnn_dim = len(train_idx)  # Number of training samples

    # Calculate distance matrix for GNN
    dist_matrix_train = distance_matrix(X_train_scaled, X_train_scaled)
    dist_matrix_test = distance_matrix(X_test_scaled, X_train_scaled)

    # Use distances to nearest neighbors as GNN features
    gnn_train_data = np.hstack([
        np.argsort(d)[:5] for d in dist_matrix_train
    ]).astype(float)  # 5 nearest neighbors for each training sample

    gnn_test_data = np.hstack([
        np.argsort(d)[:5] for d in dist_matrix_test
    ]).astype(float)  # 5 nearest neighbors from training set

    # Raster generator
    raster_gen = RasterDataGenerator(raster_paths, coords, patch_size=100)
    cnn_shape = (100, 100, len(raster_paths))

    # Model configurations
    models_config = [
        {
            "name": "1. Transformer CNN GNN MLP",
            "builder": lambda: build_transformer_model(cnn_shape, gnn_test_data.shape[1], mlp_dim),
            "has_cnn": True
        },
        {
            "name": "2. GNN MLP AE",
            "builder": lambda: build_gnn_mlp_ae_model(mlp_dim, gnn_test_data.shape[1]),
            "has_cnn": False
        },
        {
            "name": "3. CNN GNN MLP PG",
            "builder": lambda: build_cnn_gnn_mlp_pg_model(cnn_shape, gnn_test_data.shape[1], mlp_dim),
            "has_cnn": True
        },
        {
            "name": "4. GNN MLP",
            "builder": lambda: build_gnn_mlp_model(mlp_dim, gnn_test_data.shape[1]),
            "has_cnn": False
        },
        {
            "name": "5. Stacked CNN GNN MLP",
            "builder": lambda: build_stacked_cnn_gnn_mlp_model(cnn_shape, gnn_test_data.shape[1], mlp_dim),
            "has_cnn": True
        },
    ]

    # Run all models
    results = []
    for config in models_config:
        print("\n" + "="*80)
        print(config["name"])
        print("="*80)

        model = config["builder"]()
        print("Model built successfully")
        print("Training...")

        # Prepare training data
        if config["has_cnn"]:
            cnn_patches = raster_gen.get_patches(train_idx)
            history = model.fit(
                [cnn_patches, X_train_scaled, gnn_train_data],
                y_train,
                epochs=100,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)]
            )
        else:
            history = model.fit(
                [X_train_scaled, gnn_train_data],
                y_train,
                epochs=100,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)]
            )

        print(f"Training complete ({len(history.history['loss'])} epochs)")
        print("Evaluating...")

        # Evaluate
        y_pred = evaluate_model(model, X_test_scaled, gnn_test_data, y_test, raster_gen, test_idx, config["has_cnn"])
        r2, rmse, mae, smape = calculate_metrics(y_test, y_pred)

        print(f"\nResults:")
        print(f"  R²:    {r2:.4f}")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  MAE:   {mae:.4f}")
        print(f"  SMAPE: {smape:.2f}%")

        results.append({
            'Model': config["name"],
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'SMAPE': smape
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL MODELS (BASELINE)")
    print("="*80)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save results
    output_file = PROJECT_ROOT / "TOP5_RAINY_BASELINE_RESULTS.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
