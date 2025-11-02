"""
Run object detection and tracking evaluation pipeline.

Usage:
    # Run with config file
    python scripts/evaluation/run_evaluation.py \
        --config scripts/evaluation/configs/rtdetr_example.yaml \
        --sequence ARG_FRA_183303 \
        --output-dir results/ARG_FRA_183303

    # Run with specific image directory
    python scripts/evaluation/run_evaluation.py \
        --config scripts/evaluation/configs/rtdetr_example.yaml \
        --image-dir data/images/ARG_FRA_183303 \
        --sequence ARG_FRA_183303 \
        --output-dir results/ARG_FRA_183303
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from evaluation.evaluation_pipeline import EvaluationPipeline


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run object detection and tracking evaluation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Sequence name"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Path to image directory (default: data/images/<sequence>)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to output directory (default: results/<sequence>)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory (default: data/)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save output video"
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Don't save prediction file"
    )

    args = parser.parse_args()

    # Setup paths
    if args.data_dir is None:
        # Assume we're in project root or scripts/evaluation
        current = Path.cwd()
        if current.name == 'evaluation':
            data_dir = current.parent.parent / "data"
        elif current.name == 'scripts':
            data_dir = current.parent / "data"
        else:
            data_dir = current / "data"
    else:
        data_dir = args.data_dir

    if args.image_dir is None:
        image_dir = data_dir / "images" / args.sequence
    else:
        image_dir = args.image_dir

    if args.output_dir is None:
        output_dir = Path("results") / args.sequence
    else:
        output_dir = args.output_dir

    # Check if image directory exists
    if not image_dir.exists():
        print(f"❌ Error: Image directory not found: {image_dir}")
        sys.exit(1)

    # Check if config exists
    if not args.config.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run pipeline
    pipeline = EvaluationPipeline(config_path=args.config)

    results = pipeline.run(
        image_dir=image_dir,
        sequence_name=args.sequence,
        output_dir=output_dir,
        save_video=not args.no_video,
        save_predictions=not args.no_predictions
    )

    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
