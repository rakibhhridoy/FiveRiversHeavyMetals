#!/usr/bin/env python3
"""
Google Earth Engine Script to Export AlphaEarth Data
Run this to download AlphaEarth 64-dimensional embeddings as GeoTIFF
"""

import ee
import time

def export_alphaearth_data():
    """
    Export AlphaEarth embeddings from Google Earth Engine
    """

    print("="*70)
    print("ALPHAEARTH DATA EXPORT FROM GOOGLE EARTH ENGINE")
    print("="*70)

    # Initialize Earth Engine with your project
    try:
        ee.Initialize(project='five-rivers-alphaearth')
        print("✓ Earth Engine initialized with project: five-rivers-alphaearth")
    except Exception as e:
        print(f"✗ Failed to initialize Earth Engine: {e}")
        print("\nMake sure you have authenticated:")
        print("  python3 -m earthengine authenticate")
        return

    print("\n" + "-"*70)
    print("STEP 1: Define Study Area")
    print("-"*70)

    # Define study area - Dhaka, Bangladesh (Five Rivers region)
    # Adjust bounds to match your specific study area
    bounds = {
        'north': 24.0,
        'south': 23.5,
        'east': 90.0,
        'west': 88.0
    }

    aoi = ee.Geometry.Rectangle([
        bounds['west'], bounds['south'],
        bounds['east'], bounds['north']
    ])

    print(f"✓ Study Area (AOI):")
    print(f"  North: {bounds['north']}")
    print(f"  South: {bounds['south']}")
    print(f"  East: {bounds['east']}")
    print(f"  West: {bounds['west']}")

    print("\n" + "-"*70)
    print("STEP 2: Load AlphaEarth Dataset")
    print("-"*70)

    try:
        # Load AlphaEarth ImageCollection
        # AlphaEarth provides annual embeddings at 10m resolution
        embeddings_collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

        print(f"✓ AlphaEarth dataset loaded")
        print(f"  Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        print(f"  Resolution: 10m")
        print(f"  Bands: 64 dimensions")
        print(f"  Type: Float32")

        # Get the latest available image
        latest_image = embeddings_collection.first()
        print(f"✓ Latest available image selected")

    except Exception as e:
        print(f"✗ Failed to load AlphaEarth dataset: {e}")
        return

    print("\n" + "-"*70)
    print("STEP 3: Prepare for Export")
    print("-"*70)

    # Clip to study area
    clipped_image = latest_image.clip(aoi)

    print(f"✓ Image clipped to study area")

    print("\n" + "-"*70)
    print("STEP 4: Export Options")
    print("-"*70)

    export_config = {
        'description': 'AlphaEarth_64_Bands_Full',
        'folder': 'Five_Rivers',
        'scale': 10,  # 10m resolution
        'crs': 'EPSG:4326',  # WGS84
        'maxPixels': 1e13,
        'fileFormat': 'GeoTIFF',
        'formatOptions': {
            'cloudOptimized': True  # Cloud-optimized GeoTIFF
        }
    }

    print(f"Export Configuration:")
    print(f"  Description: {export_config['description']}")
    print(f"  Output Folder (Google Drive): {export_config['folder']}")
    print(f"  Resolution: {export_config['scale']}m")
    print(f"  CRS: {export_config['crs']}")
    print(f"  Format: {export_config['fileFormat']}")

    print("\n" + "-"*70)
    print("STEP 5: Start Export")
    print("-"*70)

    try:
        # Create export task for GeoTIFF
        task = ee.batch.Export.image.toDrive(
            image=clipped_image,
            description=export_config['description'],
            folder=export_config['folder'],
            scale=export_config['scale'],
            crs=export_config['crs'],
            maxPixels=int(export_config['maxPixels']),
            fileFormat=export_config['fileFormat']
        )

        # Start the task
        task.start()

        print(f"✓ Export task started!")
        print(f"  Task ID: {task.id}")
        print(f"  Description: {export_config['description']}")

        # Monitor task status
        print("\n" + "-"*70)
        print("STEP 6: Monitor Progress")
        print("-"*70)

        while True:
            status = task.status()
            state = status['state']

            print(f"  Status: {state}")

            if state == 'COMPLETED':
                print(f"\n✓ EXPORT SUCCESSFUL!")
                print(f"  Task ID: {task.id}")
                break
            elif state == 'FAILED':
                print(f"\n✗ EXPORT FAILED")
                print(f"  Error: {status.get('error_message', 'Unknown error')}")
                break
            elif state == 'CANCELLED':
                print(f"\n✗ EXPORT CANCELLED")
                break

            # Wait before checking again
            time.sleep(5)

    except Exception as e:
        print(f"✗ Failed to create export task: {e}")
        print("\nTroubleshooting:")
        print("1. Verify you have Google Drive access")
        print("2. Check project permissions")
        print("3. Ensure Earth Engine API is enabled")
        return

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. Go to: https://console.cloud.google.com/storage
   (or check Google Drive for 'Five_Rivers' folder)

2. Download 'AlphaEarth_64_Bands_Full.tif' to:
   /Users/rakibhhridoy/Five_Rivers/gis/AlphaEarth/

3. Run visualization script:
   python3 /Users/rakibhhridoy/Five_Rivers/ALPHAEARTH_VISUALIZE.py

4. View visualizations in:
   /Users/rakibhhridoy/Five_Rivers/gis/visualizations/
    """)
    print("="*70)

def export_alphaearth_by_year(start_year=2017, end_year=2024):
    """
    Export AlphaEarth data for multiple years (time series)
    """

    print("\n" + "="*70)
    print("ALPHAEARTH TIME SERIES EXPORT (OPTIONAL)")
    print("="*70)

    try:
        ee.Initialize(project='five-rivers-alphaearth')
    except:
        print("✗ Earth Engine not initialized")
        return

    # Define study area
    aoi = ee.Geometry.Rectangle([88.0, 23.5, 90.0, 24.0])

    # Load collection
    embeddings_collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

    print(f"\nExporting AlphaEarth data for years {start_year}-{end_year}")
    print("-"*70)

    tasks = []

    for year in range(start_year, end_year + 1):
        try:
            # Filter by year
            yearly_image = embeddings_collection.filterDate(
                f'{year}-01-01', f'{year}-12-31'
            ).first()

            if yearly_image is None:
                print(f"✗ No data available for {year}")
                continue

            # Clip to study area
            clipped = yearly_image.clip(aoi)

            # Create export task
            task = ee.batch.Export.image.toDrive(
                image=clipped,
                description=f'AlphaEarth_{year}',
                folder='Five_Rivers',
                scale=10,
                crs='EPSG:4326',
                maxPixels=1e13
            )

            task.start()
            tasks.append((year, task.id))

            print(f"✓ {year}: Export started (Task ID: {task.id})")

        except Exception as e:
            print(f"✗ {year}: Failed to start export - {e}")

    print("\n" + "="*70)
    print(f"Started {len(tasks)} export tasks")
    print("="*70)

    for year, task_id in tasks:
        print(f"  {year}: {task_id}")

# Main execution
if __name__ == '__main__':
    import sys

    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  ALPHAEARTH DATA EXPORT FROM GOOGLE EARTH ENGINE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    # Option 1: Export latest AlphaEarth data
    print("\nOption 1: Export latest AlphaEarth data")
    print("-"*70)
    export_alphaearth_data()

    # Option 2 (Optional): Export time series
    # Uncomment to export multiple years
    # print("\n\nOption 2: Export AlphaEarth time series (2017-2024)")
    # export_alphaearth_by_year(2017, 2024)

    print("\n✓ Script completed\n")
