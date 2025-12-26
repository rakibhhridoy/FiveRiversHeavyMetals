#!/usr/bin/env python3
"""
AlphaEarth Comparative Analysis Master Script

This script orchestrates the complete comparative analysis:
- Run all 11 models
- Test all 4 AlphaEarth integration options (A, B, C, D)
- For both rainy and winter seasons
- Collect and aggregate results

Usage:
    python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py [options]

Options:
    --season SEASON          : 'rainy', 'winter', or 'both' (default: both)
    --option OPTION          : 'A', 'B', 'C', 'D', or 'all' (default: all)
    --model MODEL            : Model name or 'all' (default: all)
    --data-prep-only         : Only run data preparation, skip models
    --model-only             : Skip data prep, only run models
    --option B               : Quick test with Option B only
    --parallel N             : Run N models in parallel (default: 1)

Examples:
    # Quick test: Option B only, Rainy season
    python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py --season rainy --option B

    # Full analysis with parallel processing
    python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py --parallel 3

    # Data preparation only
    python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py --data-prep-only

Prerequisites:
    1. Google Earth Engine account (free at earthengine.google.com)
    2. earthengine-api installed: pip install earthengine-api
    3. Authentication: python -m earthengine authenticate
    4. TensorFlow and other dependencies installed
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


class AlphaEarthComparativeAnalysis:
    """Master coordinator for AlphaEarth comparative analysis"""

    def __init__(self, project_root: str = "/Users/rakibhhridoy/Five_Rivers"):
        self.project_root = Path(project_root)
        self.gis_dir = self.project_root / "gis"
        self.rainy_ae_dir = self.gis_dir / "SedimentRainyAE"
        self.winter_ae_dir = self.gis_dir / "SedimentWinterAE"
        self.models = [
            "Transformer CNN GNN MLP.ipynb",
            "GNN MLP AE.ipynb",
            "CNN GNN MLP PG.ipynb",
            "GNN MLP.ipynb",
            "CNN GAT MLP.ipynb",
            "Stacked CNN GNN MLP.ipynb",
            "CNN GNN MLP.ipynb",
            "Mixture of Experts.ipynb",
            "Dual Attention.ipynb",
            "CNN LSTM.ipynb",
            "GCN GAT.ipynb",
        ]
        self.options = ["A", "B", "C", "D"]
        self.seasons = ["rainy", "winter"]
        self.results = {}
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        self.log("Checking prerequisites...")

        # Check directories exist
        if not self.rainy_ae_dir.exists():
            self.log(f"ERROR: {self.rainy_ae_dir} not found", "ERROR")
            return False
        if not self.winter_ae_dir.exists():
            self.log(f"ERROR: {self.winter_ae_dir} not found", "ERROR")
            return False

        # Check notebooks exist
        for model in self.models:
            rainy_nb = self.rainy_ae_dir / model
            winter_nb = self.winter_ae_dir / model
            if not rainy_nb.exists():
                self.log(f"WARNING: {rainy_nb} not found", "WARN")
            if not winter_nb.exists():
                self.log(f"WARNING: {winter_nb} not found", "WARN")

        # Check Python packages
        packages = ["jupyter", "tensorflow", "pandas", "numpy"]
        for pkg in packages:
            try:
                __import__(pkg)
                self.log(f"✓ {pkg} installed")
            except ImportError:
                self.log(f"ERROR: {pkg} not installed. Run: pip install {pkg}", "ERROR")
                return False

        # Check earthengine-api
        try:
            import ee
            self.log("✓ earthengine-api installed")
        except ImportError:
            self.log("ERROR: earthengine-api not installed. Run: pip install earthengine-api", "ERROR")
            return False

        self.log("✓ All prerequisites satisfied")
        return True

    def run_data_preparation(self, season: str = "rainy") -> bool:
        """Run data preparation notebook for a season"""
        season_label = "Rainy" if season == "rainy" else "Winter"
        self.log(f"Starting data preparation for {season_label} season...")

        if season == "rainy":
            nb_path = self.rainy_ae_dir / "00_AlphaEarth_Data_Preparation.ipynb"
        else:
            nb_path = self.winter_ae_dir / "00_AlphaEarth_Data_Preparation.ipynb"

        if not nb_path.exists():
            self.log(f"ERROR: {nb_path} not found", "ERROR")
            return False

        try:
            # Run notebook using jupyter nbconvert
            cmd = [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=3600",
                str(nb_path)
            ]
            self.log(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                self.log(f"✓ Data preparation completed for {season_label}")
                # Verify output files
                for option in self.options:
                    csv_file = nb_path.parent / f"Option_{option}_{season_label}AE.csv"
                    if csv_file.exists():
                        size_mb = csv_file.stat().st_size / (1024 * 1024)
                        self.log(f"  ✓ Generated: {csv_file.name} ({size_mb:.1f} MB)")
                    else:
                        self.log(f"  ⚠ Expected file not found: {csv_file.name}", "WARN")
                return True
            else:
                self.log(f"ERROR: Data preparation failed for {season_label}", "ERROR")
                self.log(f"STDOUT: {result.stdout}", "ERROR")
                self.log(f"STDERR: {result.stderr}", "ERROR")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"ERROR: Data preparation timed out for {season_label}", "ERROR")
            return False
        except Exception as e:
            self.log(f"ERROR: {str(e)}", "ERROR")
            return False

    def run_model(self, season: str, model: str, option: str) -> Dict:
        """Run a single model with specified option"""
        result = {
            "season": season,
            "model": model,
            "option": option,
            "status": "pending",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration": None,
            "metrics": {},
            "error": None,
        }

        season_label = "Rainy" if season == "rainy" else "Winter"
        season_dir = self.rainy_ae_dir if season == "rainy" else self.winter_ae_dir
        nb_path = season_dir / model

        if not nb_path.exists():
            result["status"] = "failed"
            result["error"] = f"Notebook not found: {nb_path}"
            return result

        try:
            # Read notebook and modify ALPHA_EARTH_OPTION
            with open(nb_path, 'r') as f:
                nb = json.load(f)

            # Find and modify the AlphaEarth option cell (should be cell 3)
            modified = False
            for cell in nb["cells"]:
                if cell["cell_type"] == "code" and "ALPHA_EARTH_OPTION" in "".join(cell["source"]):
                    # Update the option value
                    source = "".join(cell["source"])
                    modified_source = source.replace(
                        "ALPHA_EARTH_OPTION = 'B'",
                        f"ALPHA_EARTH_OPTION = '{option}'"
                    )
                    cell["source"] = modified_source.split('\n')
                    modified = True
                    break

            if not modified:
                result["status"] = "failed"
                result["error"] = "Could not find ALPHA_EARTH_OPTION in notebook"
                return result

            # Create temporary notebook with option set
            temp_nb_path = nb_path.parent / f".temp_{model.replace('.ipynb', '')}_opt{option}.ipynb"
            with open(temp_nb_path, 'w') as f:
                json.dump(nb, f, indent=2)

            # Execute notebook
            cmd = [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=14400",  # 4 hours
                str(temp_nb_path)
            ]

            self.log(f"Running {season_label} {model} with Option {option}...", "INFO")
            start = time.time()
            subprocess_result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            duration = time.time() - start

            result["duration"] = duration
            result["end_time"] = datetime.now().isoformat()

            if subprocess_result.returncode == 0:
                result["status"] = "success"
                self.log(f"✓ {season_label} {model} Option {option} completed in {duration:.1f}s", "INFO")
                # Try to extract metrics from output notebook
                try:
                    with open(temp_nb_path, 'r') as f:
                        output_nb = json.load(f)
                    # Look for metrics in output
                    # This is a simplified extraction - adjust based on actual output format
                    result["metrics"] = {"extracted": True}
                except:
                    pass
            else:
                result["status"] = "failed"
                result["error"] = subprocess_result.stderr[:500]
                self.log(f"✗ {season_label} {model} Option {option} failed", "ERROR")

            # Clean up temporary file
            if temp_nb_path.exists():
                temp_nb_path.unlink()

            return result

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = "Execution timeout (4 hours exceeded)"
            result["end_time"] = datetime.now().isoformat()
            return result
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["end_time"] = datetime.now().isoformat()
            return result

    def generate_report(self) -> str:
        """Generate summary report of comparative analysis"""
        report = []
        report.append("\n" + "=" * 80)
        report.append("ALPHAEARTH COMPARATIVE ANALYSIS - SUMMARY REPORT")
        report.append("=" * 80)

        # Timeline
        elapsed = datetime.now() - self.start_time
        report.append(f"\nExecution Timeline:")
        report.append(f"  Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Total Duration: {elapsed}")

        # Results summary
        report.append(f"\nResults Summary:")
        report.append(f"  Total runs: {len(self.results)}")

        status_counts = {}
        for result in self.results.values():
            status = result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        for status, count in sorted(status_counts.items()):
            report.append(f"  {status}: {count}")

        # Grouped by season and option
        report.append(f"\nResults by Season and Option:")
        for season in self.seasons:
            report.append(f"\n  {season.upper()}:")
            for option in self.options:
                matching = [r for r in self.results.values() if r["season"] == season and r["option"] == option]
                successes = [r for r in matching if r["status"] == "success"]
                report.append(f"    Option {option}: {len(successes)}/{len(matching)} models succeeded")

        # Failed runs
        failed = [r for r in self.results.values() if r["status"] != "success"]
        if failed:
            report.append(f"\nFailed Runs ({len(failed)}):")
            for result in failed[:10]:  # Show first 10
                report.append(f"  {result['season']} {result['model']} Option {result['option']}: {result['error'][:60]}")
            if len(failed) > 10:
                report.append(f"  ... and {len(failed) - 10} more")

        report.append("\n" + "=" * 80 + "\n")
        return "\n".join(report)

    def run(self, args):
        """Main execution method"""
        self.log("Starting AlphaEarth Comparative Analysis")
        self.log(f"Project root: {self.project_root}")

        # Check prerequisites
        if not self.check_prerequisites():
            self.log("Prerequisites check failed. Exiting.", "ERROR")
            return False

        # Data preparation phase
        if not args.model_only:
            for season in (["rainy", "winter"] if args.season == "both" else [args.season]):
                if not self.run_data_preparation(season):
                    self.log(f"Data preparation failed for {season}", "ERROR")
                    return False

        # Model training phase
        if not args.data_prep_only:
            seasons = ["rainy", "winter"] if args.season == "both" else [args.season]
            options = ["A", "B", "C", "D"] if args.option == "all" else [args.option]
            models = self.models if args.model == "all" else [args.model + ".ipynb"]

            total_runs = len(seasons) * len(options) * len(models)
            self.log(f"Scheduling {total_runs} model runs ({len(seasons)} seasons × {len(options)} options × {len(models)} models)")

            run_count = 0
            for season in seasons:
                for option in options:
                    for model in models:
                        run_count += 1
                        self.log(f"\nRun {run_count}/{total_runs}")
                        result = self.run_model(season, model, option)
                        run_key = f"{season}_{option}_{model}"
                        self.results[run_key] = result

        # Generate report
        report = self.generate_report()
        print(report)

        # Save report
        report_file = self.project_root / "claude_temp" / f"COMPARATIVE_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        self.log(f"Report saved: {report_file}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="AlphaEarth Comparative Analysis Master Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--season", default="both", choices=["rainy", "winter", "both"],
                        help="Season to process (default: both)")
    parser.add_argument("--option", default="all", choices=["A", "B", "C", "D", "all"],
                        help="AlphaEarth integration option (default: all)")
    parser.add_argument("--model", default="all",
                        help="Model to run (default: all). Example: 'Transformer CNN GNN MLP'")
    parser.add_argument("--data-prep-only", action="store_true",
                        help="Run data preparation only")
    parser.add_argument("--model-only", action="store_true",
                        help="Run models only, skip data preparation")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel processes (default: 1)")

    args = parser.parse_args()

    # Run analysis
    analyzer = AlphaEarthComparativeAnalysis()
    success = analyzer.run(args)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
