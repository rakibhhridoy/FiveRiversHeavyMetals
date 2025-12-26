#!/usr/bin/env python3
"""
AlphaEarth Data Visualization Script
Converts 64-dimensional embeddings to viewable raster visualizations
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AlphaEarthVisualizer:
    def __init__(self, raster_path, output_dir):
        self.raster_path = Path(raster_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_raster(self):
        """Load AlphaEarth raster file"""
        with rasterio.open(self.raster_path) as src:
            self.data = src.read()
            self.profile = src.profile
            self.h, self.w = self.data.shape[1], self.data.shape[2]

        print(f"✓ Loaded {self.profile['count']} bands, {self.h}×{self.w} resolution")

    def visualize_all_bands(self):
        """Create 8×8 grid of all 64 bands"""
        print("Generating 64-band grid visualization...")

        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        axes = axes.flatten()

        for band_idx in range(1, 65):
            band_data = self.data[band_idx - 1]

            # Normalize using percentiles
            p2, p98 = np.percentile(band_data, [2, 98])
            norm_data = (band_data - p2) / (p98 - p2 + 1e-8)

            axes[band_idx - 1].imshow(np.clip(norm_data, 0, 1), cmap='viridis')
            axes[band_idx - 1].set_title(f'Band {band_idx}', fontsize=8)
            axes[band_idx - 1].axis('off')

        plt.tight_layout()
        output_path = self.output_dir / 'all_64_bands.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def visualize_pca_composite(self):
        """Create RGB composite from 3 principal components"""
        print("Generating PCA composite (RGB)...")

        # Reshape data for PCA
        data_reshaped = self.data.transpose(1, 2, 0).reshape(-1, 64)

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)

        # Apply PCA
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(data_scaled)
        pca_image = pca_data.reshape(self.h, self.w, 3)

        # Normalize to 0-1
        pca_normalized = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min() + 1e-8)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(pca_normalized)
        explained_var = sum(pca.explained_variance_ratio_)
        ax.set_title(f'AlphaEarth PCA Composite (64→3 dimensions)\nExplained Variance: {explained_var:.1%}')
        ax.axis('off')

        output_path = self.output_dir / 'pca_composite.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")
        print(f"  Explained variance: {explained_var:.2%}")
        print(f"  - PC1: {pca.explained_variance_ratio_[0]:.2%}")
        print(f"  - PC2: {pca.explained_variance_ratio_[1]:.2%}")
        print(f"  - PC3: {pca.explained_variance_ratio_[2]:.2%}")

    def save_pca_raster(self):
        """Save PCA-reduced data as GeoTIFF (3 bands)"""
        print("Saving PCA raster as GeoTIFF...")

        # Reshape and prepare data
        data_reshaped = self.data.transpose(1, 2, 0).reshape(-1, 64)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)

        # Apply PCA
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(data_scaled)
        pca_image = pca_data.reshape(self.h, self.w, 3)

        # Update profile and save
        output_path = self.output_dir / 'AlphaEarth_PCA3.tif'
        profile = self.profile.copy()
        profile.update(count=3, dtype=rasterio.float32)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pca_image.transpose(2, 0, 1))

        print(f"✓ Saved: {output_path.name}")
        print(f"  Original: 64 bands")
        print(f"  Reduced: 3 bands (PC1, PC2, PC3)")

    def band_statistics(self):
        """Analyze and visualize band statistics"""
        print("Computing band statistics...")

        band_means = []
        band_stds = []
        band_mins = []
        band_maxs = []

        for i in range(64):
            band = self.data[i]
            band_means.append(np.mean(band))
            band_stds.append(np.std(band))
            band_mins.append(np.min(band))
            band_maxs.append(np.max(band))

        # Plot statistics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mean
        axes[0, 0].bar(range(1, 65), band_means, color='steelblue')
        axes[0, 0].set_title('Mean Values per Band')
        axes[0, 0].set_xlabel('Band')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].grid(True, alpha=0.3)

        # Std Dev
        axes[0, 1].bar(range(1, 65), band_stds, color='darkorange')
        axes[0, 1].set_title('Standard Deviation per Band')
        axes[0, 1].set_xlabel('Band')
        axes[0, 1].set_ylabel('Std Dev')
        axes[0, 1].grid(True, alpha=0.3)

        # Min/Max Range
        axes[1, 0].fill_between(range(1, 65), band_mins, band_maxs, alpha=0.3, color='green')
        axes[1, 0].plot(range(1, 65), band_means, color='darkgreen', linewidth=2)
        axes[1, 0].set_title('Min/Max Range with Mean')
        axes[1, 0].set_xlabel('Band')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)

        # Distribution
        axes[1, 1].hist(band_means, bins=20, color='purple', alpha=0.7, label='Mean')
        axes[1, 1].hist(band_stds, bins=20, color='orange', alpha=0.7, label='Std Dev')
        axes[1, 1].set_title('Distribution of Band Statistics')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'band_statistics.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def band_correlation(self):
        """Analyze band correlations"""
        print("Computing band correlation matrix...")

        # Reshape for correlation
        data_reshaped = self.data.transpose(1, 2, 0).reshape(-1, 64)
        correlation = np.corrcoef(data_reshaped.T)

        # Plot full correlation matrix
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(correlation, cmap='coolwarm', center=0, square=True,
                    xticklabels=False, yticklabels=False, ax=ax,
                    cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('AlphaEarth Band Correlation Matrix (All 64 Bands)')
        plt.tight_layout()

        output_path = self.output_dir / 'band_correlation_full.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plot subset for clarity
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation[:16, :16], cmap='coolwarm', center=0,
                    xticklabels=[f'B{i+1}' for i in range(16)],
                    yticklabels=[f'B{i+1}' for i in range(16)],
                    ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('AlphaEarth Band Correlation (First 16 Bands)')
        plt.tight_layout()

        output_path = self.output_dir / 'band_correlation_subset.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved correlation matrices")

    def visualize_individual_bands(self, band_indices=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50]):
        """Create detailed visualization of selected bands"""
        print(f"Visualizing selected bands: {band_indices}")

        num_bands = len(band_indices)
        cols = 5
        rows = (num_bands + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, band_num in enumerate(band_indices):
            if band_num < 1 or band_num > 64:
                continue

            band_data = self.data[band_num - 1]

            # Normalize
            p2, p98 = np.percentile(band_data, [2, 98])
            norm_data = (band_data - p2) / (p98 - p2 + 1e-8)

            im = axes[idx].imshow(np.clip(norm_data, 0, 1), cmap='viridis')
            axes[idx].set_title(f'Band {band_num}')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(len(band_indices), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = self.output_dir / 'selected_bands.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {output_path.name}")

    def run_all(self):
        """Execute all visualizations"""
        print("\n" + "="*70)
        print("ALPHAEARTH VISUALIZATION PIPELINE")
        print("="*70)

        self.load_raster()
        print()

        self.visualize_all_bands()
        print()

        self.visualize_pca_composite()
        print()

        self.save_pca_raster()
        print()

        self.band_statistics()
        print()

        self.band_correlation()
        print()

        self.visualize_individual_bands()
        print()

        print("="*70)
        print("VISUALIZATION COMPLETE")
        print("="*70)
        print(f"\nAll visualizations saved to:")
        print(f"  {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  - {file.name}")
        print("\n" + "="*70)

# Main execution
if __name__ == '__main__':
    import sys

    # Default paths (modify as needed)
    raster_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'
    output_dir = '/Users/rakibhhridoy/Five_Rivers/gis/visualizations/'

    # Check if file exists
    if not Path(raster_path).exists():
        print(f"✗ AlphaEarth raster not found: {raster_path}")
        print("\nTo get AlphaEarth data:")
        print("1. Run the Google Earth Engine export script")
        print("2. Download AlphaEarth_64_Bands.tif from Google Drive")
        print("3. Place in: /Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/")
        sys.exit(1)

    # Run visualizer
    visualizer = AlphaEarthVisualizer(raster_path, output_dir)
    visualizer.run_all()
