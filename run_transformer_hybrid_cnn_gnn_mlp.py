#!/usr/bin/env python3
"""
Hybrid Transformer CNN GNN MLP - Simplified for Small Datasets
Uses basic feature concatenation instead of complex attention
Focus on efficient feature extraction and simple fusion
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import distance_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import rasterio
from rasterio.windows import Window
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/Users/rakibhhridoy/Five_Rivers")
GIS_ROOT = PROJECT_ROOT / "gis" / "SedimentRainyAE"

RAINY_DATA_FILE = GIS_ROOT / "Option_B_RainyAE_SELECTED.csv"
SELECTED_FEATURES_FILE = GIS_ROOT / "Option_B_RainyAE_SELECTED_FEATURES.txt"
RASTER_CAL_INDICES = GIS_ROOT.parent / "CalIndices" / "*.tif"
RASTER_LULC = GIS_ROOT.parent / "LULCMerged" / "*.tif"
RASTER_IDW = GIS_ROOT.parent / "IDW" / "*.tif"

BATCH_SIZE = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 80
PATIENCE = 12

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load AlphaEarth data with selected features"""
    print("\n" + "="*80)
    print("LOADING DATA - HYBRID TRANSFORMER MODEL")
    print("="*80)

    with open(SELECTED_FEATURES_FILE) as f:
        selected_features = [line.strip() for line in f.readlines()]

    print(f"Loading: {RAINY_DATA_FILE}")
    data = pd.read_csv(RAINY_DATA_FILE)

    feature_cols = selected_features
    X = data[feature_cols].values.astype(float)
    y = data['RI'].values.astype(float)
    coords = data[['Long', 'Lat']].values

    print(f"✓ Loaded {len(data)} samples with {len(feature_cols)} features")
    print(f"  Target range: {y.min():.2f} - {y.max():.2f}")

    return X, y, coords, feature_cols

def load_rasters():
    """Load raster files"""
    print("\nLoading raster layers...")
    raster_paths = []
    raster_paths += glob.glob(str(RASTER_CAL_INDICES))
    raster_paths += glob.glob(str(RASTER_LULC))
    raster_paths += glob.glob(str(RASTER_IDW))
    print(f"✓ Found {len(raster_paths)} raster layers")
    return sorted(raster_paths)

class RasterDataGenerator:
    """Generate CNN patches from rasters"""
    def __init__(self, raster_files, sample_coords, patch_size=100):
        self.raster_files = raster_files
        self.coords = sample_coords
        self.patch_size = patch_size

    def get_patches(self, sample_indices):
        """Extract CNN patches"""
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
# HYBRID TRANSFORMER MODEL
# ============================================================================

def build_hybrid_transformer_model(cnn_shape, gnn_dim, mlp_dim):
    """
    Hybrid Transformer with Simple Concatenation Fusion
    - Efficient CNN for spatial features
    - Minimal MLPand GNN branches
    - Simple concatenation fusion (proven to work on small datasets)
    - No complex attention mechanisms
    """

    # =========== CNN BRANCH ===========
    cnn_input = Input(shape=cnn_shape, name='cnn_input')

    # Efficient 2-block CNN
    x_cnn = Conv2D(16, 3, padding='same', activation='relu')(cnn_input)
    x_cnn = Conv2D(16, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = MaxPooling2D(2)(x_cnn)
    x_cnn = Dropout(0.2)(x_cnn)

    x_cnn = Conv2D(32, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = Conv2D(32, 3, padding='same', activation='relu')(x_cnn)
    x_cnn = GlobalAveragePooling2D()(x_cnn)
    x_cnn = Dropout(0.2)(x_cnn)

    x_cnn = Dense(64, activation='relu')(x_cnn)
    x_cnn = Dropout(0.3)(x_cnn)

    # =========== MLP BRANCH ===========
    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')

    x_mlp = Dense(64, activation='relu')(mlp_input)
    x_mlp = Dropout(0.2)(x_mlp)
    x_mlp = Dense(32, activation='relu')(x_mlp)
    x_mlp = Dropout(0.2)(x_mlp)

    # =========== GNN BRANCH ===========
    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')

    x_gnn = Dense(32, activation='relu')(gnn_input)
    x_gnn = Dropout(0.2)(x_gnn)
    x_gnn = Dense(16, activation='relu')(x_gnn)
    x_gnn = Dropout(0.2)(x_gnn)

    # =========== SIMPLE CONCATENATION FUSION ===========
    # Instead of complex attention, use proven simple concatenation
    x_fused = Concatenate()([x_cnn, x_mlp, x_gnn])

    # Fusion dense layers (simple, stable processing)
    x_fused = Dense(128, activation='relu')(x_fused)
    x_fused = Dropout(0.3)(x_fused)
    x_fused = Dense(64, activation='relu')(x_fused)
    x_fused = Dropout(0.2)(x_fused)
    x_fused = Dense(32, activation='relu')(x_fused)
    x_fused = Dropout(0.2)(x_fused)

    # Output layer
    output = Dense(1)(x_fused)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)

    # Conservative learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    smape = np.mean(diff[denominator != 0]) * 100 if np.any(denominator != 0) else 0

    return r2, rmse, mae, smape

def evaluate_model(model, mlp_data, gnn_data, y_true, raster_gen=None, test_indices=None):
    """Evaluate model"""
    cnn_patches = raster_gen.get_patches(test_indices)
    y_pred = model.predict([cnn_patches, mlp_data, gnn_data], verbose=0)
    return y_pred.flatten()

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train_scaled, y_train, gnn_train_data,
               raster_gen, train_idx, validation_split=0.2):
    """Train model with stable configuration"""
    print("Training hybrid transformer with simple concatenation fusion...")

    cnn_patches = raster_gen.get_patches(train_idx)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.8,
        patience=8,
        min_lr=0.0001,
        verbose=1
    )

    history = model.fit(
        [cnn_patches, X_train_scaled, gnn_train_data],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=validation_split,
        callbacks=[early_stop, reduce_lr]
    )

    return history

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("HYBRID TRANSFORMER CNN GNN MLP FOR ALPHAEARTH")
    print("="*80)

    # Load data
    X, y, coords, feature_cols = load_data()
    raster_paths = load_rasters()

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

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nPreparing GNN features...")
    dist_matrix_train = distance_matrix(X_train_scaled, X_train_scaled)
    dist_matrix_test = distance_matrix(X_test_scaled, X_train_scaled)
    gnn_train_data = dist_matrix_train.mean(axis=1, keepdims=True)
    gnn_test_data = dist_matrix_test.mean(axis=1, keepdims=True)

    print(f"MLP features: {X_train_scaled.shape[1]}")
    print(f"GNN dimension: {len(train_idx)}")
    print(f"CNN patch shape: (100, 100, {len(raster_paths)})")

    # Raster generator
    raster_gen = RasterDataGenerator(raster_paths, coords, patch_size=100)
    cnn_shape = (100, 100, len(raster_paths))

    # Build model
    print("\n" + "="*80)
    print("BUILDING HYBRID TRANSFORMER MODEL")
    print("="*80)

    model = build_hybrid_transformer_model(cnn_shape, gnn_test_data.shape[1], X_train_scaled.shape[1])
    print("✓ Model built successfully")
    print(model.summary())

    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    history = train_model(
        model, X_train_scaled, y_train, gnn_train_data,
        raster_gen, train_idx
    )

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    y_pred = evaluate_model(model, X_test_scaled, gnn_test_data, y_test, raster_gen, test_idx)
    r2, rmse, mae, smape = calculate_metrics(y_test, y_pred)

    print(f"\nHybrid Transformer CNN GNN MLP Results:")
    print(f"  R²:    {r2:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  SMAPE: {smape:.2f}%")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS VERSIONS")
    print("="*80)

    print("\nResults Summary:")
    print(f"  20 features (unoptimized Transformer):  R² = 0.7672")
    print(f"  20 features (GNN MLP AE - BEST):        R² = 0.9725")
    print(f"  20 features (Hybrid Concatenation):     R² = {r2:.4f}")

    if r2 > 0.7672:
        improvement = (r2 - 0.7672) / 0.7672 * 100
        print(f"\n  ✓ Improvement over original Transformer: +{improvement:.1f}%")
    if r2 > 0.9:
        print(f"  ✓✓ EXCELLENT: Approaching best GNN MLP AE performance!")

    # Save results
    output_file = PROJECT_ROOT / "TRANSFORMER_HYBRID_RESULTS.csv"
    results = pd.DataFrame({
        'Model': ['Transformer CNN GNN MLP (Hybrid Concatenation)'],
        'R²': [r2],
        'RMSE': [rmse],
        'MAE': [mae],
        'SMAPE': [smape]
    })
    results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
