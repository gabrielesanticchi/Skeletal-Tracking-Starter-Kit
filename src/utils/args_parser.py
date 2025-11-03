"""
Common argument parser for visualization scripts.

Provides reusable CLI argument parsing to avoid repetition.
"""

import argparse
from pathlib import Path
from typing import Optional


class ArgsParser:
    """
    Common argument parser for visualization and processing scripts.

    Provides standard arguments for sequence selection, frame selection,
    data directory, and output options.
    """

    @staticmethod
    def create_base_parser(description: str) -> argparse.ArgumentParser:
        """
        Create base parser with common arguments.

        Args:
            description: Script description

        Returns:
            ArgumentParser with common arguments
        """
        parser = argparse.ArgumentParser(description=description)

        parser.add_argument(
            '--sequence',
            type=str,
            default=None,
            help='Sequence name (e.g., ARG_FRA_183303). If not specified, random sequence is selected.'
        )

        parser.add_argument(
            '--frame',
            type=int,
            default=None,
            help='Frame index. If not specified, random frame is selected.'
        )

        parser.add_argument(
            '--data-dir',
            type=Path,
            default=None,
            help='Base data directory (default: auto-detect from project root)'
        )

        parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='Output file path to save visualization. If not specified, displays interactively.'
        )

        return parser

    @staticmethod
    def add_3d_viz_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add 3D visualization specific arguments.

        Args:
            parser: Existing argument parser

        Returns:
            Parser with 3D viz arguments added
        """
        parser.add_argument(
            '--elev',
            type=int,
            default=0,
            help='Elevation angle for 3D plot (default: 0)'
        )

        parser.add_argument(
            '--azim',
            type=int,
            default=-90,
            help='Azimuth angle for 3D plot (default: -90)'
        )

        parser.add_argument(
            '--figsize',
            type=int,
            nargs=2,
            default=[15, 5],
            help='Figure size (width height) (default: 15 5)'
        )

        parser.add_argument(
            '--num-subjects',
            type=int,
            default=None,
            help='Number of subjects to plot (default: None, plots all subjects)'
        )

        return parser

    @staticmethod
    def add_animation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add animation specific arguments.

        Args:
            parser: Existing argument parser

        Returns:
            Parser with animation arguments added
        """
        parser.add_argument(
            '--start-frame',
            type=int,
            default=0,
            help='Starting frame for animation (default: 0)'
        )

        parser.add_argument(
            '--end-frame',
            type=int,
            default=None,
            help='Ending frame for animation (default: None, uses last frame)'
        )

        parser.add_argument(
            '--frame-step',
            type=int,
            default=1,
            help='Frame step size for animation (default: 1)'
        )

        parser.add_argument(
            '--fps',
            type=int,
            default=10,
            help='Frames per second for animation (default: 10)'
        )

        parser.add_argument(
            '--duration',
            type=float,
            default=None,
            help='Animation duration in seconds (overrides fps if specified)'
        )

        return parser

    @staticmethod
    def add_2d_viz_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add 2D visualization specific arguments.

        Args:
            parser: Existing argument parser

        Returns:
            Parser with 2D viz arguments added
        """
        parser.add_argument(
            '--show-skeleton',
            action='store_true',
            default=True,
            help='Show skeleton connections (default: True)'
        )

        parser.add_argument(
            '--show-joints',
            action='store_true',
            default=True,
            help='Show joint points (default: True)'
        )

        parser.add_argument(
            '--show-labels',
            action='store_true',
            default=False,
            help='Show joint labels (default: False)'
        )

        parser.add_argument(
            '--num-subjects',
            type=int,
            default=None,
            help='Number of subjects to plot (default: None, plots all subjects)'
        )

        return parser

    @staticmethod
    def get_project_root() -> Path:
        """
        Get the project root directory.

        Returns:
            Path to project root
        """
        # Assumes this file is in src/utils/
        return Path(__file__).resolve().parent.parent.parent

    @staticmethod
    def get_data_dir(args: argparse.Namespace) -> Path:
        """
        Get data directory from args or default.

        Args:
            args: Parsed arguments

        Returns:
            Path to data directory
        """
        return args.data_dir or ArgsParser.get_project_root() / 'data'
