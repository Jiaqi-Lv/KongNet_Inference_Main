#!/usr/bin/env python3
"""
KongNet Output to QuPath Converter

This script converts KongNet inference outputs (SQLite .db files) to GeoJSON format
for visualization in QuPath. It processes cell detection annotations and converts
them to point annotations with appropriate cell type classifications and colors.

Usage:
    python output_to_qupath.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]

Example:
    python output_to_qupath.py --input-dir KongNet_Outputs --output-dir QuPath_annotations

The script expects .db files with naming convention: {WSI_ID}_{suffix}.db
and creates organized output directories for each WSI.

"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from tiatoolbox.annotation import SQLiteStore
except ImportError:
    print("Error: tiatoolbox is required. Install with: pip install tiatoolbox")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KongNetToQuPathConverter:
    """
    Converter class for transforming KongNet outputs to QuPath-compatible GeoJSON format.

    This class handles the conversion of cell detection results stored in SQLite databases
    to GeoJSON format that can be imported into QuPath for visualization and analysis.
    """

    # Standard cell type color mapping for QuPath visualization
    # Colors are in RGB format: (R, G, B)
    # This example is for CoNIC. You can modify these colors as needed.
    DEFAULT_COLOR_MAPPING: Dict[str, Tuple[int, int, int]] = {
        "neutrophil": (255, 0, 0),  # Red
        "lymphocyte": (0, 255, 0),  # Green
        "plasma": (0, 0, 255),  # Blue
        "epithelial": (255, 255, 0),  # Yellow
        "connective": (0, 255, 255),  # Cyan
        "eosinophil": (255, 0, 255),  # Magenta
        "unknown": (128, 128, 128),  # Gray (fallback)
    }

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        """
        Initialize the converter.

        Args:
            input_dir (str): Directory containing KongNet .db output files
            output_dir (str): Directory where GeoJSON files will be saved
            color_mapping (Optional[Dict]): Custom color mapping for cell types.
                                          If None, uses default mapping.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.color_mapping = color_mapping or self.DEFAULT_COLOR_MAPPING

        # Validate directories
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized converter: {input_dir} -> {output_dir}")

    def extract_wsi_id(self, filename: str) -> str:
        """
        Extract WSI ID from annotation filename.

        Assumes naming convention: {WSI_ID}_{suffix}.db

        Args:
            filename (str): The annotation filename

        Returns:
            str: Extracted WSI ID
        """
        name_without_ext = Path(filename).stem
        wsi_id = "_".join(name_without_ext.split("_")[:-1])
        return wsi_id if wsi_id else name_without_ext

    def generate_random_id(self) -> str:
        """
        Generate a random hexadecimal ID for GeoJSON features.

        Returns:
            str: 24-character hexadecimal string
        """
        return "".join(random.choice("0123456789abcdef") for _ in range(24))

    def convert_annotation_to_geojson_feature(
        self, annotation, feature_id: str
    ) -> Dict:
        """
        Convert a single annotation to a GeoJSON feature.

        Args:
            annotation: TIAToolbox annotation object
            feature_id (str): Unique ID for the feature

        Returns:
            Dict: GeoJSON feature dictionary
        """
        cell_type = annotation.properties.get("type", "unknown")
        poly = annotation.geometry
        centroid = poly.centroid

        # Get color for cell type, fallback to unknown if not found
        color = self.color_mapping.get(cell_type, self.color_mapping["unknown"])

        feature = {
            "type": "Feature",
            "id": feature_id,
            "properties": {
                "objectType": "annotation",
                "name": cell_type,
                "classification": {
                    "name": cell_type,
                    "color": color,
                },
            },
            "geometry": {"type": "Point", "coordinates": [centroid.x, centroid.y]},
        }

        return feature

    def convert_db_to_geojson(self, db_path: Path, output_path: Path) -> bool:
        """
        Convert a single .db file to GeoJSON format.

        Args:
            db_path (Path): Path to input .db file
            output_path (Path): Path for output .geojson file

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Open the annotation store
            annotation_store = SQLiteStore.open(str(db_path))

            # Initialize GeoJSON structure
            geojson_dict = {"type": "FeatureCollection", "features": []}

            # Convert each annotation to a GeoJSON feature
            for annotation in annotation_store.values():
                feature_id = self.generate_random_id()
                feature = self.convert_annotation_to_geojson_feature(
                    annotation, feature_id
                )
                geojson_dict["features"].append(feature)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write GeoJSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson_dict, f, indent=4)

            logger.info(
                f"Converted {len(geojson_dict['features'])} annotations to {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Error converting {db_path}: {str(e)}")
            return False

    def convert_all(self, skip_existing: bool = True) -> Dict[str, bool]:
        """
        Convert all .db files in the input directory to GeoJSON format.

        Args:
            skip_existing (bool): Whether to skip files that already exist

        Returns:
            Dict[str, bool]: Mapping of filenames to conversion success status
        """
        # Find all .db files
        db_files = list(self.input_dir.glob("*.db"))

        if not db_files:
            logger.warning(f"No .db files found in {self.input_dir}")
            return {}

        logger.info(f"Found {len(db_files)} .db files to process")

        results = {}

        for db_file in db_files:
            # Extract WSI ID and create output structure
            wsi_id = self.extract_wsi_id(db_file.name)
            output_wsi_dir = self.output_dir / wsi_id
            output_file = output_wsi_dir / "KongNet_CoNIC.geojson"

            logger.info(f"Processing {db_file.name} for WSI ID: {wsi_id}")

            # Skip if file already exists and skip_existing is True
            if output_file.exists() and skip_existing:
                logger.info(f"Skipping {db_file.name}, output already exists.")
                results[db_file.name] = True
                continue

            # Perform conversion
            success = self.convert_db_to_geojson(db_file, output_file)
            results[db_file.name] = success

        return results

    def print_summary(self, results: Dict[str, bool]) -> None:
        """
        Print a summary of conversion results.

        Args:
            results (Dict[str, bool]): Conversion results
        """
        total = len(results)
        successful = sum(results.values())
        failed = total - successful

        print(f"\n{'='*50}")
        print("CONVERSION SUMMARY")
        print(f"{'='*50}")
        print(f"Total files processed: {total}")
        print(f"Successful conversions: {successful}")
        print(f"Failed conversions: {failed}")

        if failed > 0:
            print(f"\nFailed files:")
            for filename, success in results.items():
                if not success:
                    print(f"  - {filename}")

        print(f"{'='*50}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert KongNet .db outputs to QuPath-compatible GeoJSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir KongNet_Outputs --output-dir QuPath_annotations
  %(prog)s --input-dir ./results --output-dir ./qupath --force
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="KongNet_Outputs",
        help="Directory containing KongNet .db output files (default: KongNet_Outputs)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="QuPath_annotations",
        help="Directory to save QuPath .geojson files (default: QuPath_annotations)",
    )

    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing output files"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def main():
    """Main function to run the converter."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize converter
        converter = KongNetToQuPathConverter(
            input_dir=args.input_dir, output_dir=args.output_dir
        )

        # Run conversion
        results = converter.convert_all(skip_existing=not args.force)

        # Print summary
        converter.print_summary(results)

        # Exit with appropriate code
        failed_count = sum(1 for success in results.values() if not success)
        sys.exit(0 if failed_count == 0 else 1)

    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
