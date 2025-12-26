# AlphaEarth Data Visualization Guide
## Converting 64-Dimensional Embeddings to Raster Files

---

## Overview

AlphaEarth provides 64-dimensional satellite embeddings that are difficult to visualize directly. This guide shows how to:
1. Extract AlphaEarth data from Google Earth Engine
2. Convert to raster files (GeoTIFF format)
3. Visualize individual bands
4. Create composite visualizations
5. Generate PCA-reduced visualizations

---

## Method 1: Extract AlphaEarth as Raster from Google Earth Engine

### Step 1: Basic Script to Download AlphaEarth Raster

```python
import ee
import geemap
import json

# Initialize Earth Engine
ee.Initialize(project='five-rivers-alphaearth')

# Define study area (Dhaka, Bangladesh - adjust to your area)
aoi = ee.Geometry.Rectangle([88.0, 23.5, 90.0, 24.0])

# Load AlphaEarth dataset
embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').first()

# Export all 64 bands to GeoTIFF
task = ee.batch.Export.image.toDrive(
    image=embeddings.clip(aoi),
    description='AlphaEarth_64_Bands',
    folder='Five_Rivers',
    scale=10,  # 10m resolution (native AlphaEarth resolution)
    crs='EPSG:4326',
    maxPixels=1e13,
    fileFormat='GeoTIFF'
)
task.start()
print(f"Export task started: {task.id}")

# Check task status
task.status()
```

### Step 2: Download from Google Drive

Once exported, download the GeoTIFF file:
- File will be in your Google Drive in the `Five_Rivers` folder
- Name: `AlphaEarth_64_Bands.tif`
- Size: ~500MB-1GB (64 bands × full raster)
- Save to: `/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/`

---

## Method 2: Visualize Individual Bands

### Step 1: Load and Inspect Bands

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# Load AlphaEarth raster
raster_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'

with rasterio.open(raster_path) as src:
    # Get metadata
    print(f"Number of bands: {src.count}")
    print(f"Resolution: {src.res}")
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")

    # Read specific band (e.g., band 1)
    band_1 = src.read(1)

    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(band_1, cmap='viridis')
    plt.title('AlphaEarth Band 1')
    plt.colorbar(label='Embedding Value')
    plt.savefig('/Users/rakibhhridoy/Five_Rivers/gis/visualizations/band_1.png', dpi=150, bbox_inches='tight')
    plt.close()

print("✓ Band visualization saved")
```

### Step 2: Visualize All 64 Bands

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

raster_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'

with rasterio.open(raster_path) as src:
    # Create 8×8 grid for 64 bands
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    axes = axes.flatten()

    for band_idx in range(1, 65):
        band_data = src.read(band_idx)

        # Normalize for visualization
        norm = Normalize(vmin=np.percentile(band_data, 2),
                        vmax=np.percentile(band_data, 98))

        # Plot
        im = axes[band_idx - 1].imshow(band_data, cmap='viridis', norm=norm)
        axes[band_idx - 1].set_title(f'Band {band_idx}', fontsize=8)
        axes[band_idx - 1].axis('off')

    plt.tight_layout()
    plt.savefig('/Users/rakibhhridoy/Five_Rivers/gis/visualizations/all_64_bands.png',
                dpi=150, bbox_inches='tight')
    plt.close()

print("✓ All 64 bands visualization saved")
```

---

## Method 3: PCA-Based Visualization (RGB Composite)

### Reduce 64 dimensions to 3 principal components for RGB visualization

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

raster_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'

with rasterio.open(raster_path) as src:
    # Read all 64 bands
    data = src.read()  # Shape: (64, height, width)

    # Reshape for PCA: (height*width, 64)
    h, w = data.shape[1], data.shape[2]
    data_reshaped = data.transpose(1, 2, 0).reshape(-1, 64)

    # Apply PCA to reduce to 3 components (RGB)
    print("Applying PCA...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data_scaled)

    # Reshape back to image
    pca_image = pca_data.reshape(h, w, 3)

    # Normalize to 0-255 for visualization
    pca_normalized = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min()) * 255

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    # Visualize PCA composite
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Original first 3 bands
    rgb_original = data[:3].transpose(1, 2, 0)
    rgb_normalized = (rgb_original - rgb_original.min()) / (rgb_original.max() - rgb_original.min())
    ax1.imshow(rgb_normalized)
    ax1.set_title('Original First 3 AlphaEarth Bands')
    ax1.axis('off')

    # PCA reduced
    ax2.imshow(pca_normalized.astype(np.uint8))
    ax2.set_title('PCA Composite (3 Principal Components)')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/rakibhhridoy/Five_Rivers/gis/visualizations/pca_composite.png',
                dpi=150, bbox_inches='tight')
    plt.close()

print("✓ PCA composite saved")
```

---

## Method 4: Save PCA-Reduced Raster (3-Band GeoTIFF)

### Create a new GeoTIFF with just 3 PCA bands

```python
import rasterio
from rasterio.transform import Affine
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

input_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'
output_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_PCA3.tif'

with rasterio.open(input_path) as src:
    # Read all 64 bands
    data = src.read()  # Shape: (64, height, width)
    h, w = data.shape[1], data.shape[2]

    # Prepare data for PCA
    data_reshaped = data.transpose(1, 2, 0).reshape(-1, 64)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)

    # Apply PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data_scaled)
    pca_image = pca_data.reshape(h, w, 3)

    # Save as GeoTIFF
    profile = src.profile
    profile.update(count=3, dtype=rasterio.float32)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pca_image.transpose(2, 0, 1))

print(f"✓ PCA raster saved: {output_path}")
print(f"  - Original: 64 bands")
print(f"  - Reduced: 3 bands (PC1, PC2, PC3)")
print(f"  - Explained variance: {sum(pca.explained_variance_ratio_):.2%}")
```

---

## Method 5: Band Statistics and Correlation

### Analyze AlphaEarth bands

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

raster_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'

with rasterio.open(raster_path) as src:
    data = src.read()  # Shape: (64, height, width)
    h, w = data.shape[1], data.shape[2]

    # Calculate statistics for each band
    band_stats = []
    for i in range(1, 65):
        band = src.read(i)
        band_stats.append({
            'Band': i,
            'Mean': np.mean(band),
            'Std': np.std(band),
            'Min': np.min(band),
            'Max': np.max(band)
        })

    # Reshape for correlation
    data_reshaped = data.transpose(1, 2, 0).reshape(-1, 64)
    correlation = np.corrcoef(data_reshaped.T)

    # Plot correlation heatmap (sample of bands)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation[:16, :16], cmap='coolwarm', center=0,
                xticklabels=[f'B{i+1}' for i in range(16)],
                yticklabels=[f'B{i+1}' for i in range(16)],
                ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('AlphaEarth Band Correlation Matrix (First 16 Bands)')
    plt.tight_layout()
    plt.savefig('/Users/rakibhhridoy/Five_Rivers/gis/visualizations/band_correlation.png',
                dpi=150, bbox_inches='tight')
    plt.close()

print("✓ Band statistics and correlation computed")
```

---

## Method 6: Interactive Visualization with Leaflet/Folium

### Create interactive map to explore bands

```python
import rasterio
import numpy as np
import folium
from folium.plugins import ImageOverlay
import json

raster_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'

with rasterio.open(raster_path) as src:
    # Get bounds
    bounds = src.bounds
    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # Read first band for overlay
    band_1 = src.read(1)
    band_1_normalized = (band_1 - band_1.min()) / (band_1.max() - band_1.min())

    # Note: This is a simplified example
    # Full implementation requires converting raster to web-friendly format

    m.save('/Users/rakibhhridoy/Five_Rivers/gis/visualizations/alphaearth_map.html')

print("✓ Interactive map created")
```

---

## Method 7: Time-Series Visualization (If available)

### If you have AlphaEarth data for multiple years

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming you have multiple AlphaEarth files for different years
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
base_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/'

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, year in enumerate(years):
    file_path = Path(base_path) / f'AlphaEarth_{year}.tif'

    if file_path.exists():
        with rasterio.open(file_path) as src:
            band_1 = src.read(1)

            im = axes[idx].imshow(band_1, cmap='viridis')
            axes[idx].set_title(f'Year {year}')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.suptitle('AlphaEarth Time Series (Band 1)', fontsize=14)
plt.tight_layout()
plt.savefig('/Users/rakibhhridoy/Five_Rivers/gis/visualizations/timeseries.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("✓ Time series visualization created")
```

---

## Complete Visualization Script

### All-in-one script to generate all visualizations

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def visualize_alphaearth(raster_path, output_dir):
    """
    Generate all AlphaEarth visualizations
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading AlphaEarth raster: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read()  # Shape: (64, height, width)
        h, w = data.shape[1], data.shape[2]

        print(f"✓ Loaded {src.count} bands, {h}×{w} resolution")

        # 1. Individual bands (8×8 grid)
        print("Generating 64-band grid visualization...")
        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        axes = axes.flatten()

        for band_idx in range(1, 65):
            band_data = src.read(band_idx)
            norm_data = (band_data - np.percentile(band_data, 2)) / \
                       (np.percentile(band_data, 98) - np.percentile(band_data, 2))

            axes[band_idx - 1].imshow(np.clip(norm_data, 0, 1), cmap='viridis')
            axes[band_idx - 1].set_title(f'Band {band_idx}', fontsize=8)
            axes[band_idx - 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'all_64_bands.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: all_64_bands.png")

        # 2. PCA Composite (RGB)
        print("Generating PCA composite...")
        data_reshaped = data.transpose(1, 2, 0).reshape(-1, 64)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)

        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(data_scaled)
        pca_image = pca_data.reshape(h, w, 3)
        pca_normalized = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())

        plt.figure(figsize=(12, 10))
        plt.imshow(pca_normalized)
        plt.title(f'PCA Composite (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})')
        plt.axis('off')
        plt.savefig(output_dir / 'pca_composite.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: pca_composite.png")

        # 3. Band statistics
        print("Computing band statistics...")
        band_means = []
        band_stds = []
        for i in range(1, 65):
            band = src.read(i)
            band_means.append(np.mean(band))
            band_stds.append(np.std(band))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(range(1, 65), band_means, marker='o', markersize=4)
        ax1.set_xlabel('Band')
        ax1.set_ylabel('Mean Value')
        ax1.set_title('AlphaEarth Band Means')
        ax1.grid(True, alpha=0.3)

        ax2.plot(range(1, 65), band_stds, marker='o', markersize=4, color='orange')
        ax2.set_xlabel('Band')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('AlphaEarth Band Standard Deviations')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'band_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: band_statistics.png")

        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"Output directory: {output_dir}")
        print(f"Files generated:")
        print(f"  - all_64_bands.png (8×8 grid of all bands)")
        print(f"  - pca_composite.png (RGB from 64 dimensions)")
        print(f"  - band_statistics.png (mean and std plots)")
        print("="*60)

# Usage
if __name__ == '__main__':
    raster_path = '/Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/AlphaEarth_64_Bands.tif'
    output_dir = '/Users/rakibhhridoy/Five_Rivers/gis/visualizations/'

    visualize_alphaearth(raster_path, output_dir)
```

---

## Summary: Quick Start

### **To visualize AlphaEarth data, follow these steps:**

1. **Export from Google Earth Engine**
   ```bash
   # Run Python script with ee.batch.Export command
   # Download AlphaEarth_64_Bands.tif from Google Drive
   ```

2. **Run visualization script**
   ```bash
   python3 /Users/rakibhhridoy/Five_Rivers/ALPHAEARTH_VISUALIZE.py
   ```

3. **View outputs**
   - `all_64_bands.png` - Individual band inspection
   - `pca_composite.png` - RGB composite from 64 dimensions
   - `band_statistics.png` - Distribution analysis

---

## File Sizes & Performance

| Method | File Size | Processing Time | Use Case |
|--------|-----------|-----------------|----------|
| All 64 Bands (GeoTIFF) | 500MB-1GB | 2-3 min | Full analysis |
| PCA 3-Band (GeoTIFF) | ~25MB | 1 min | Fast visualization |
| PNG Visualizations | 5-50MB | 30s | Sharing results |
| Individual Bands | 8-15MB each | <1s | Band inspection |

---

## Recommendations

✅ **Use PCA composite** for quick visualization (3 principal components)
✅ **Use full 64 bands** for detailed analysis
✅ **Create GeoTIFF** for integration with GIS software (ArcGIS, QGIS)
✅ **Standardize bands** before analysis (StandardScaler)
✅ **Use percentile clipping** for visualization (2nd to 98th percentile)

---

## Next Steps

1. Export AlphaEarth from Google Earth Engine
2. Run visualization script
3. Examine PCA composite for feature patterns
4. Use individual bands for correlation analysis
5. Create custom composites based on your analysis needs
