#!/usr/bin/env python3
"""
Run Top 5 Rainy Season Models with AlphaEarth - Single Train/Test Split
Simple direct execution without notebooks
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import distance_matrix
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, MultiHeadAttention, LayerNormalization, Reshape, LSTM, GRU, Bidirectional, RepeatVector, TimeDistributed, Attention
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
TEST_SIZE = 0.2
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

# ============================================================================
# DATA GENERATOR
# ============================================================================

def extract_patch_for_generator(coords, raster_files, buffer_pixels_x, buffer_pixels_y, patch_width, patch_height):
    """Extract CNN patches from rasters"""
    patches = []
    for lon, lat in coords:
        channels = []
        for rfile in raster_files:
            try:
                with rasterio.open(rfile) as src:
                    row, col = src.index(lon, lat)
                    win = Window(col - buffer_pixels_x, row - buffer_pixels_y, patch_width, patch_height)
                    arr = src.read(1, window=win, boundless=True, fill_value=0)
                    arr = arr.astype(np.float32)
                    if np.nanmax(arr) != 0:
                        arr /= np.nanmax(arr)
            except Exception as e:
                arr = np.zeros((patch_width, patch_height), dtype=np.float32)
            channels.append(arr)
        patches.append(np.stack(channels, axis=-1))

    return np.array(patches)

class DataGenerator(Sequence):
    def __init__(self, coords, mlp_data, gnn_data, y, raster_paths, buffer_meters, batch_size=4, shuffle=True):
        self.coords = coords
        self.mlp_data = mlp_data
        self.gnn_data = gnn_data
        self.y = y
        self.raster_paths = raster_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))

        with rasterio.open(raster_paths[0]) as src:
            res_x, res_y = src.res
            self.buffer_pixels_x = int(buffer_meters / res_x)
            self.buffer_pixels_y = int(buffer_meters / res_y)
            self.patch_width = 2 * self.buffer_pixels_x
            self.patch_height = 2 * self.buffer_pixels_y

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_coords = self.coords[batch_indices]
        batch_mlp = self.mlp_data[batch_indices]
        batch_gnn = self.gnn_data[batch_indices, :]
        batch_y = self.y[batch_indices]

        batch_cnn = extract_patch_for_generator(
            batch_coords, self.raster_paths, self.buffer_pixels_x, self.buffer_pixels_y,
            self.patch_width, self.patch_height
        )

        return (batch_cnn, batch_mlp, batch_gnn), batch_y

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def build_transformer_model(cnn_shape, gnn_dim, mlp_dim):
    """Transformer CNN GNN MLP Fusion Model"""
    cnn_input = Input(shape=cnn_shape, name="cnn_input")
    mlp_input = Input(shape=(mlp_dim,), name="mlp_input")
    gnn_input = Input(shape=(gnn_dim,), name="gnn_input")

    # CNN
    cnn = Conv2D(32, (3,3), activation="relu", padding="same")(cnn_input)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Conv2D(64, (3,3), activation="relu", padding="same")(cnn)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Flatten(name="cnn_embedding")(cnn)

    # MLP
    mlp = Dense(128, activation="relu")(mlp_input)
    mlp = Dense(64, activation="relu", name="mlp_embedding")(mlp)

    # GNN
    gnn = Dense(128, activation="relu")(gnn_input)
    gnn = Dense(64, activation="relu", name="gnn_embedding")(gnn)

    # Transformer Fusion
    projection_dim = 64
    cnn_proj = Dense(projection_dim)(cnn)
    mlp_proj = Dense(projection_dim)(mlp)
    gnn_proj = Dense(projection_dim)(gnn)

    cnn_exp = Reshape((1, projection_dim))(cnn_proj)
    mlp_exp = Reshape((1, projection_dim))(mlp_proj)
    gnn_exp = Reshape((1, projection_dim))(gnn_proj)
    embeddings = Concatenate(axis=1)([cnn_exp, mlp_exp, gnn_exp])

    transformer = MultiHeadAttention(num_heads=4, key_dim=projection_dim)(embeddings, embeddings)
    transformer = Dropout(0.2)(transformer)
    transformer = LayerNormalization(epsilon=1e-6)(embeddings + transformer)
    transformer = Flatten()(transformer)

    # Output
    f = Dense(128, activation="relu")(transformer)
    f = Dropout(0.4)(f)
    f = Dense(64, activation="relu")(f)
    output = Dense(1, activation="linear")(f)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
    return model

def build_gnn_mlp_ae_model(gnn_dim, mlp_dim):
    """GNN MLP AutoEncoder Model"""
    gnn_input = Input(shape=(gnn_dim,), name="gnn_input")
    mlp_input = Input(shape=(mlp_dim,), name="mlp_input")

    gnn = Dense(64, activation="relu")(gnn_input)
    gnn = Dense(32, activation="relu")(gnn)
    mlp = Dense(64, activation="relu")(mlp_input)
    mlp = Dense(32, activation="relu")(mlp)

    combined = Concatenate()([gnn, mlp])
    x = Dense(64, activation="relu")(combined)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[gnn_input, mlp_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

def build_cnn_gnn_mlp_pg_model(cnn_shape, gnn_dim, mlp_dim):
    """CNN GNN MLP Progressive Model"""
    cnn_input = Input(shape=cnn_shape, name="cnn_input")
    mlp_input = Input(shape=(mlp_dim,), name="mlp_input")
    gnn_input = Input(shape=(gnn_dim,), name="gnn_input")

    cnn = Conv2D(32, (3,3), activation="relu")(cnn_input)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(64, activation="relu", name="cnn_embedding")(cnn)

    mlp = Dense(64, activation="relu")(mlp_input)
    mlp = Dense(32, activation="relu", name="mlp_embedding")(mlp)

    gnn = Dense(64, activation="relu")(gnn_input)
    gnn = Dense(32, activation="relu", name="gnn_embedding")(gnn)

    combined = Concatenate()([cnn, mlp, gnn])
    x = Dense(128, activation="relu")(combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
    return model

def build_gnn_mlp_model(gnn_dim, mlp_dim):
    """Simple GNN MLP Model"""
    gnn_input = Input(shape=(gnn_dim,), name="gnn_input")
    mlp_input = Input(shape=(mlp_dim,), name="mlp_input")

    gnn = Dense(128, activation="relu")(gnn_input)
    gnn = Dense(64, activation="relu")(gnn)

    mlp = Dense(128, activation="relu")(mlp_input)
    mlp = Dense(64, activation="relu")(mlp)

    combined = Concatenate()([gnn, mlp])
    x = Dense(128, activation="relu")(combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[gnn_input, mlp_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

def build_stacked_cnn_gnn_mlp_model(cnn_shape, gnn_dim, mlp_dim):
    """Stacked CNN GNN MLP Model"""
    cnn_input = Input(shape=cnn_shape, name="cnn_input")
    mlp_input = Input(shape=(mlp_dim,), name="mlp_input")
    gnn_input = Input(shape=(gnn_dim,), name="gnn_input")

    cnn = Conv2D(64, (3,3), activation="relu", padding="same")(cnn_input)
    cnn = Conv2D(32, (3,3), activation="relu", padding="same")(cnn)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(128, activation="relu")(cnn)
    cnn = Dense(64, activation="relu", name="cnn_embedding")(cnn)

    mlp = Dense(128, activation="relu")(mlp_input)
    mlp = Dense(64, activation="relu")(mlp)
    mlp = Dense(32, activation="relu", name="mlp_embedding")(mlp)

    gnn = Dense(128, activation="relu")(gnn_input)
    gnn = Dense(64, activation="relu")(gnn)
    gnn = Dense(32, activation="relu", name="gnn_embedding")(gnn)

    stacked = Concatenate()([cnn, mlp, gnn])
    x = Dense(256, activation="relu")(stacked)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs=[cnn_input, mlp_input, gnn_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
    return model

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def evaluate_model(model, coords, mlp_data, gnn_data, y_true, raster_paths, batch_size=4, has_cnn=True):
    """Evaluate model and return predictions"""
    y_pred_list = []
    num_samples = len(y_true)

    with rasterio.open(raster_paths[0]) as src:
        res_x, res_y = src.res
        buffer_pixels_x = int(BUFFER_METERS / res_x)
        buffer_pixels_y = int(BUFFER_METERS / res_y)
        patch_width = 2 * buffer_pixels_x
        patch_height = 2 * buffer_pixels_y

    for i in range(0, num_samples, batch_size):
        batch_coords = coords[i:i+batch_size]
        batch_mlp = mlp_data[i:i+batch_size]
        batch_gnn = gnn_data[i:i+batch_size, :]

        if has_cnn:
            batch_cnn = extract_patch_for_generator(
                batch_coords, raster_paths, buffer_pixels_x, buffer_pixels_y, patch_width, patch_height
            )
            preds = model.predict((batch_cnn, batch_mlp, batch_gnn), verbose=0)
        else:
            preds = model.predict((batch_gnn, batch_mlp), verbose=0)

        y_pred_list.append(preds.flatten())

    return np.concatenate(y_pred_list)

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return {"R²": r2, "RMSE": rmse, "MAE": mae, "SMAPE": smape}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("RUNNING TOP 5 RAINY SEASON MODELS WITH ALPHAEARTH")
    print("Single Train/Test Split (80/20)")
    print("="*80)

    # Load data
    X, y, coords, feature_cols = load_data()
    raster_paths = load_rasters()

    # Create train/test split
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    coords_train, coords_test = coords[train_idx], coords[test_idx]

    # Prepare MLP and GNN data
    print("\nPreparing MLP/GNN data...")
    scaler = StandardScaler()
    mlp_train = scaler.fit_transform(X_train)
    mlp_test = scaler.transform(X_test)

    dist_mat_train = distance_matrix(coords_train, coords_train)
    gnn_train = np.exp(-dist_mat_train / 10)
    dist_mat_test_train = distance_matrix(coords_test, coords_train)
    gnn_test = np.exp(-dist_mat_test_train / 10)

    # Calculate CNN patch shape
    with rasterio.open(raster_paths[0]) as src:
        res_x, res_y = src.res
        buffer_pixels_x = int(BUFFER_METERS / res_x)
        patch_width = 2 * buffer_pixels_x
        cnn_patch_shape = (patch_width, patch_width, len(raster_paths))

    print(f"MLP features: {mlp_train.shape[1]}")
    print(f"GNN dimension: {gnn_train.shape[1]}")
    print(f"CNN patch shape: {cnn_patch_shape}")

    # Define models
    models_config = [
        {
            "name": "1. Transformer CNN GNN MLP",
            "builder": lambda: build_transformer_model(cnn_patch_shape, len(gnn_train), mlp_train.shape[1]),
            "has_cnn": True,
            "has_gnn": True
        },
        {
            "name": "2. GNN MLP AE",
            "builder": lambda: build_gnn_mlp_ae_model(len(gnn_train), mlp_train.shape[1]),
            "has_cnn": False,
            "has_gnn": True
        },
        {
            "name": "3. CNN GNN MLP PG",
            "builder": lambda: build_cnn_gnn_mlp_pg_model(cnn_patch_shape, len(gnn_train), mlp_train.shape[1]),
            "has_cnn": True,
            "has_gnn": True
        },
        {
            "name": "4. GNN MLP",
            "builder": lambda: build_gnn_mlp_model(len(gnn_train), mlp_train.shape[1]),
            "has_cnn": False,
            "has_gnn": True
        },
        {
            "name": "5. Stacked CNN GNN MLP",
            "builder": lambda: build_stacked_cnn_gnn_mlp_model(cnn_patch_shape, len(gnn_train), mlp_train.shape[1]),
            "has_cnn": True,
            "has_gnn": True
        }
    ]

    results_summary = []

    # Run each model
    for config in models_config:
        print("\n" + "="*80)
        print(config["name"])
        print("="*80)

        try:
            # Build model
            model = config["builder"]()
            print(f"Model built successfully")

            # Create data generators
            if config["has_cnn"]:
                train_gen = DataGenerator(coords_train, mlp_train, gnn_train, y_train, raster_paths, BUFFER_METERS, BATCH_SIZE)
                val_gen = DataGenerator(coords_test, mlp_test, gnn_test, y_test, raster_paths, BUFFER_METERS, BATCH_SIZE, shuffle=False)

            # Train
            print(f"Training...")
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            if config["has_cnn"]:
                history = model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=[early_stop], verbose=0)
            else:
                history = model.fit(
                    (gnn_train, mlp_train), y_train,
                    validation_data=((gnn_test, mlp_test), y_test),
                    epochs=100, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=0
                )

            print(f"Training complete ({len(history.history['loss'])} epochs)")

            # Evaluate
            print(f"Evaluating...")
            if config["has_cnn"]:
                y_pred = evaluate_model(model, coords_test, mlp_test, gnn_test, y_test, raster_paths, has_cnn=True)
            else:
                y_pred = model.predict((gnn_test, mlp_test), verbose=0).flatten()

            metrics = calculate_metrics(y_test, y_pred)

            print(f"\nResults:")
            print(f"  R²:    {metrics['R²']:.4f}")
            print(f"  RMSE:  {metrics['RMSE']:.4f}")
            print(f"  MAE:   {metrics['MAE']:.4f}")
            print(f"  SMAPE: {metrics['SMAPE']:.2f}%")

            results_summary.append({
                "Model": config["name"],
                "R²": metrics['R²'],
                "RMSE": metrics['RMSE'],
                "MAE": metrics['MAE'],
                "SMAPE": metrics['SMAPE']
            })

        except Exception as e:
            print(f"✗ Error: {str(e)[:200]}")
            results_summary.append({
                "Model": config["name"],
                "R²": "ERROR",
                "RMSE": "ERROR",
                "MAE": "ERROR",
                "SMAPE": "ERROR"
            })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL MODELS")
    print("="*80)
    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False))

    # Save results
    output_file = PROJECT_ROOT / "TOP5_RAINY_ALPHAEARTH_RESULTS.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
