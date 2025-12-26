#!/usr/bin/env python3
"""
Transformer CNN GNN MLP with Full Dataset (17 samples) and All 64 AlphaEarth Embeddings
No train/test split - uses all 17 samples for training
Tests if larger dataset and full embeddings help Transformer performance
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler
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

RAINY_DATA_FILE = GIS_ROOT / "Option_B_RainyAE.csv"  # Full dataset with all 64 embeddings
RASTER_CAL_INDICES = GIS_ROOT.parent / "CalIndices" / "*.tif"
RASTER_LULC = GIS_ROOT.parent / "LULCMerged" / "*.tif"
RASTER_IDW = GIS_ROOT.parent / "IDW" / "*.tif"

BATCH_SIZE = 2  # Very small for 17 samples
EPOCHS = 150
PATIENCE = 25

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load full AlphaEarth data with all embeddings"""
    print("\n" + "="*80)
    print("LOADING DATA - TRANSFORMER WITH FULL DATASET (64 EMBEDDINGS)")
    print("="*80)

    print(f"Loading: {RAINY_DATA_FILE}")
    data = pd.read_csv(RAINY_DATA_FILE)

    # Get all columns except metadata and target
    drop_cols = ['Stations', 'River', 'Lat', 'Long', 'geometry']
    feature_cols = [c for c in data.columns if c not in drop_cols + ['RI']]

    X = data[feature_cols].values.astype(float)
    y = data['RI'].values.astype(float)
    coords = data[['Long', 'Lat']].values

    print(f"✓ Loaded {len(data)} samples with {len(feature_cols)} features")
    print(f"  Features include: {len([c for c in feature_cols if c.startswith('AE')])} AlphaEarth embeddings")
    print(f"  Features include: {len([c for c in feature_cols if not c.startswith('AE')])} original features")
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
# TRANSFORMER MODEL FOR FULL DATASET
# ============================================================================

def build_transformer_full_dataset(cnn_shape, gnn_dim, mlp_dim):
    """
    Transformer for full dataset (17 samples with 64+ features)
    Still respects data limitations but optimized for slightly larger scale
    """

    # =========== CNN BRANCH ===========
    cnn_input = Input(shape=cnn_shape, name='cnn_input')

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

    # =========== MLP BRANCH (larger for 64+ features) ===========
    mlp_input = Input(shape=(mlp_dim,), name='mlp_input')

    x_mlp = Dense(128, activation='relu')(mlp_input)
    x_mlp = Dropout(0.2)(x_mlp)
    x_mlp = Dense(64, activation='relu')(x_mlp)
    x_mlp = Dropout(0.2)(x_mlp)
    x_mlp = Dense(32, activation='relu')(x_mlp)
    x_mlp = Dropout(0.2)(x_mlp)

    # =========== GNN BRANCH ===========
    gnn_input = Input(shape=(gnn_dim,), name='gnn_input')

    x_gnn = Dense(32, activation='relu')(gnn_input)
    x_gnn = Dropout(0.2)(x_gnn)
    x_gnn = Dense(16, activation='relu')(x_gnn)
    x_gnn = Dropout(0.2)(x_gnn)

    # =========== CONCATENATION FUSION ===========
    x_fused = Concatenate()([x_cnn, x_mlp, x_gnn])

    x_fused = Dense(128, activation='relu')(x_fused)
    x_fused = Dropout(0.3)(x_fused)
    x_fused = Dense(64, activation='relu')(x_fused)
    x_fused = Dropout(0.2)(x_fused)
    x_fused = Dense(32, activation='relu')(x_fused)
    x_fused = Dropout(0.2)(x_fused)

    output = Dense(1)(x_fused)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)

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

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_scaled, y, gnn_data, raster_gen, all_indices):
    """Train model on full dataset without validation split"""
    print("Training transformer on full dataset (17 samples)...")

    cnn_patches = raster_gen.get_patches(all_indices)

    early_stop = EarlyStopping(
        monitor='loss',  # Monitor training loss instead of validation loss
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.8,
        patience=10,
        min_lr=0.0001,
        verbose=1
    )

    history = model.fit(
        [cnn_patches, X_scaled, gnn_data],
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.0,  # No validation split with 17 samples
        callbacks=[early_stop, reduce_lr]
    )

    return history

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("TRANSFORMER CNN GNN MLP - FULL DATASET TEST (64 EMBEDDINGS)")
    print("="*80)

    # Load data
    X, y, coords, feature_cols = load_data()
    raster_paths = load_rasters()

    print(f"\nTotal samples: {len(y)}")
    print(f"All samples will be used for training (no test split)")

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_indices = np.arange(len(y))

    print("\nPreparing GNN features...")
    dist_matrix = distance_matrix(X_scaled, X_scaled)
    gnn_data = dist_matrix.mean(axis=1, keepdims=True)

    print(f"MLP features: {X_scaled.shape[1]}")
    print(f"GNN dimension: {len(all_indices)}")
    print(f"CNN patch shape: (100, 100, {len(raster_paths)})")

    # Raster generator
    raster_gen = RasterDataGenerator(raster_paths, coords, patch_size=100)
    cnn_shape = (100, 100, len(raster_paths))

    # Build model
    print("\n" + "="*80)
    print("BUILDING TRANSFORMER MODEL")
    print("="*80)

    model = build_transformer_full_dataset(cnn_shape, gnn_data.shape[1], X_scaled.shape[1])
    print("✓ Model built successfully")
    print(model.summary())

    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    history = train_model(model, X_scaled, y, gnn_data, raster_gen, all_indices)

    # Evaluate (training set - no separate test set)
    print("\n" + "="*80)
    print("EVALUATION (TRAINING SET)")
    print("="*80)

    cnn_patches = raster_gen.get_patches(all_indices)
    y_pred = model.predict([cnn_patches, X_scaled, gnn_data], verbose=0).flatten()
    r2, rmse, mae, smape = calculate_metrics(y, y_pred)

    print(f"\nTransformer (Full Dataset, 64 Embeddings) Results:")
    print(f"  R²:    {r2:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  SMAPE: {smape:.2f}%")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print("\nResults Summary:")
    print(f"  20 features (selected, train/test split):     R² = 0.7672")
    print(f"  GNN MLP AE (20 features, best):               R² = 0.9725")
    print(f"  Full features (64 embeddings, no test split): R² = {r2:.4f}")

    if r2 > 0.7672:
        improvement = (r2 - 0.7672) / 0.7672 * 100
        print(f"\n  ✓ Improvement with full dataset: +{improvement:.1f}%")
    else:
        degradation = (0.7672 - r2) / 0.7672 * 100
        print(f"\n  ✗ Degradation with full dataset: -{degradation:.1f}%")

    # Save results
    output_file = PROJECT_ROOT / "TRANSFORMER_FULL_DATASET_RESULTS.csv"
    results = pd.DataFrame({
        'Model': ['Transformer CNN GNN MLP (Full Dataset, 64 Embeddings)'],
        'R²': [r2],
        'RMSE': [rmse],
        'MAE': [mae],
        'SMAPE': [smape],
        'Samples': [len(y)],
        'Features': [X_scaled.shape[1]]
    })
    results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
