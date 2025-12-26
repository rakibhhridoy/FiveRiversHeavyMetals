#!/usr/bin/env python3
"""
AlphaEarth Top 5 Models - Optimized Comparative Analysis

This script runs only the top 5 best-performing models:

RAINY SEASON (by R²):
  1. Transformer CNN GNN MLP       (R² = 0.9604)
  2. GNN MLP AE                    (R² = 0.9581)
  3. CNN GNN MLP PG                (R² = 0.957)
  4. GNN MLP                       (R² = 0.9519)
  5. Stacked CNN GNN MLP           (R² = 0.924)

WINTER SEASON (by R²):
  1. Transformer CNN GNN MLP       (R² = 0.9721)
  2. GNN MLP AE                    (R² = 0.9718)
  3. Mixture of Experts            (R² = 0.97)
  4. Stacked CNN GNN MLP           (R² = 0.9685)
  5. GNN MLP                       (R² = 0.9705)

Usage:
    python3 RUN_ALPHAEARTH_TOP5.py [options]

Options:
    --season SEASON          : 'rainy', 'winter', or 'both' (default: both)
    --option OPTION          : 'B' only (Option B is recommended and fastest)
    --model MODEL            : Specific model name (default: all top 5)
    --data-prep-only         : Only run data preparation
    --model-only             : Skip data prep, only run models

Examples:
    # Recommended: Both seasons, Option B
    python3 RUN_ALPHAEARTH_TOP5.py --season both

    # Rainy season only
    python3 RUN_ALPHAEARTH_TOP5.py --season rainy

    # Test one model
    python3 RUN_ALPHAEARTH_TOP5.py --season rainy --model "Transformer CNN GNN MLP"

    # Data preparation only
    python3 RUN_ALPHAEARTH_TOP5.py --season both --data-prep-only
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class AlphaEarthTop5Analysis:
    """Optimized analyzer for top 5 models only"""

    def __init__(self, project_root: str = "/Users/rakibhhridoy/Five_Rivers"):
        self.project_root = Path(project_root)
        self.gis_dir = self.project_root / "gis"
        self.rainy_ae_dir = self.gis_dir / "SedimentRainyAE"
        self.winter_ae_dir = self.gis_dir / "SedimentWinterAE"

        # Top 5 models per season (by R² performance)
        self.rainy_top5 = [
            "Transformer CNN GNN MLP.ipynb",
            "GNN MLP AE.ipynb",
            "CNN GNN MLP PG.ipynb",
            "GNN MLP.ipynb",
            "Stacked CNN GNN MLP.ipynb",
        ]

        self.winter_top5 = [
            "Transformer CNN GNN MLP.ipynb",
            "GNN MLP AE.ipynb",
            "Mixture of Experts.ipynb",
            "Stacked CNN GNN MLP.ipynb",
            "GNN MLP.ipynb",
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

        if not self.rainy_ae_dir.exists() or not self.winter_ae_dir.exists():
            self.log("ERROR: SedimentRainyAE or SedimentWinterAE directories not found", "ERROR")
            return False

        packages = ["jupyter", "tensorflow", "pandas", "numpy"]
        for pkg in packages:
            try:
                __import__(pkg)
                self.log(f"✓ {pkg} installed")
            except ImportError:
                self.log(f"ERROR: {pkg} not installed", "ERROR")
                return False

        try:
            import ee
            self.log("✓ earthengine-api installed")
        except ImportError:
            self.log("ERROR: earthengine-api not installed", "ERROR")
            return False

        self.log("✓ All prerequisites satisfied")
        return True

    def run_data_preparation(self, season: str = "rainy") -> bool:
        """Run data preparation notebook for a season"""
        season_label = "Rainy" if season == "rainy" else "Winter"
        self.log(f"Starting data preparation for {season_label} season...")

        nb_path = (self.rainy_ae_dir if season == "rainy" else self.winter_ae_dir) / "00_AlphaEarth_Data_Preparation.ipynb"

        if not nb_path.exists():
            self.log(f"ERROR: {nb_path} not found", "ERROR")
            return False

        try:
            cmd = [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=3600",
                str(nb_path)
            ]
            self.log(f"Running data preparation...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                self.log(f"✓ Data preparation completed for {season_label}")
                for option in self.options:
                    csv_file = nb_path.parent / f"Option_{option}_{season_label}AE.csv"
                    if csv_file.exists():
                        size_mb = csv_file.stat().st_size / (1024 * 1024)
                        self.log(f"  ✓ {csv_file.name} ({size_mb:.1f} MB)")
                return True
            else:
                self.log(f"ERROR: Data preparation failed", "ERROR")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"ERROR: Data preparation timeout", "ERROR")
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
            "error": None,
        }

        season_label = "Rainy" if season == "rainy" else "Winter"
        season_dir = self.rainy_ae_dir if season == "rainy" else self.winter_ae_dir
        nb_path = season_dir / model

        if not nb_path.exists():
            result["status"] = "failed"
            result["error"] = f"Notebook not found"
            return result

        try:
            # Read and modify notebook
            with open(nb_path, 'r') as f:
                nb = json.load(f)

            # Find and modify ALPHA_EARTH_OPTION
            modified = False
            for cell in nb["cells"]:
                if cell["cell_type"] == "code" and "ALPHA_EARTH_OPTION" in "".join(cell["source"]):
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
                result["error"] = "Could not find ALPHA_EARTH_OPTION"
                return result

            # Save temporary notebook
            temp_nb_path = nb_path.parent / f".temp_{model.replace('.ipynb', '')}_opt{option}.ipynb"
            with open(temp_nb_path, 'w') as f:
                json.dump(nb, f, indent=2)

            # Execute
            cmd = [
                "python3", "-m", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=14400",
                str(temp_nb_path)
            ]

            model_name = model.replace(".ipynb", "")
            self.log(f"Running {season_label} [{model_name}] Option {option}...")

            start = time.time()
            subprocess_result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            duration = time.time() - start

            result["duration"] = duration
            result["end_time"] = datetime.now().isoformat()

            if subprocess_result.returncode == 0:
                result["status"] = "success"
                self.log(f"✓ {model_name} completed ({duration:.0f}s)")
            else:
                result["status"] = "failed"
                result["error"] = subprocess_result.stderr[:200] if subprocess_result.stderr else "Unknown error"
                self.log(f"✗ {model_name} failed", "ERROR")

            # Cleanup
            if temp_nb_path.exists():
                temp_nb_path.unlink()

            return result

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = "4 hour timeout exceeded"
            result["end_time"] = datetime.now().isoformat()
            return result
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["end_time"] = datetime.now().isoformat()
            return result

    def generate_report(self) -> str:
        """Generate summary report"""
        report = []
        report.append("\n" + "=" * 80)
        report.append("ALPHAEARTH TOP 5 MODELS - ANALYSIS REPORT")
        report.append("=" * 80)

        elapsed = datetime.now() - self.start_time
        report.append(f"\nExecution Time: {elapsed}")

        report.append(f"\nResults ({len(self.results)} total):")

        status_counts = {}
        for result in self.results.values():
            status = result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        for status, count in sorted(status_counts.items()):
            report.append(f"  {status}: {count}")

        report.append(f"\nBy Season:")
        for season in self.seasons:
            matching = [r for r in self.results.values() if r["season"] == season]
            if matching:
                successes = [r for r in matching if r["status"] == "success"]
                report.append(f"  {season.upper()}: {len(successes)}/{len(matching)} succeeded")

        failed = [r for r in self.results.values() if r["status"] != "success"]
        if failed:
            report.append(f"\nFailed Runs:")
            for result in failed[:5]:
                report.append(f"  {result['season']} {result['model']} Opt{result['option']}: {result['error'][:40]}")

        report.append("\n" + "=" * 80 + "\n")
        return "\n".join(report)

    def run(self, args):
        """Main execution"""
        self.log("Starting AlphaEarth Top 5 Models Analysis")

        if not self.check_prerequisites():
            return False

        # Data preparation
        if not args.model_only:
            for season in (["rainy", "winter"] if args.season == "both" else [args.season]):
                if not self.run_data_preparation(season):
                    self.log(f"Data prep failed for {season}", "ERROR")
                    return False

        # Model training
        if not args.data_prep_only:
            seasons = ["rainy", "winter"] if args.season == "both" else [args.season]
            options = ["A", "B", "C", "D"] if args.option == "all" else [args.option]

            # Get models for each season
            models_by_season = {
                "rainy": self.rainy_top5 if args.model == "all" else [args.model + ".ipynb"],
                "winter": self.winter_top5 if args.model == "all" else [args.model + ".ipynb"],
            }

            total_runs = sum(len(models_by_season[s]) for s in seasons) * len(options)
            self.log(f"Scheduling {total_runs} model runs (Top 5 × {len(options)} options × {len(seasons)} seasons)")

            run_count = 0
            for season in seasons:
                models = models_by_season[season]
                for option in options:
                    for model in models:
                        run_count += 1
                        result = self.run_model(season, model, option)
                        run_key = f"{season}_{option}_{model}"
                        self.results[run_key] = result

        # Report
        report = self.generate_report()
        print(report)

        report_file = self.project_root / "claude_temp" / f"ALPHAEARTH_TOP5_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        self.log(f"Report saved: {report_file}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="AlphaEarth Top 5 Models - Optimized Comparative Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--season", default="both", choices=["rainy", "winter", "both"],
                        help="Season to process (default: both)")
    parser.add_argument("--option", default="B", choices=["B"],
                        help="AlphaEarth option (B is recommended and only option)")
    parser.add_argument("--model", default="all",
                        help="Specific model (default: all top 5)")
    parser.add_argument("--data-prep-only", action="store_true",
                        help="Data preparation only")
    parser.add_argument("--model-only", action="store_true",
                        help="Models only, skip data preparation")

    args = parser.parse_args()

    analyzer = AlphaEarthTop5Analysis()
    success = analyzer.run(args)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
