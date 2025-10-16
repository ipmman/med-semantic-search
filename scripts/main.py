#!/usr/bin/env python3
"""
Execution entry point for the CDS2016 experiment program.

This script ensures that the project path is set up correctly and executes the main CLI application.
"""
import sys
import logging
from pathlib import Path
from typing import NoReturn


def setup_project_path() -> Path:
    """Sets up the project path and returns the project root directory.
    
    Returns:
        Path: A Path object for the project root directory.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def main() -> NoReturn:
    """Executes the main program.
    
    Sets up the project path and then runs the main CLI program for CDS2016.
    Handles any unexpected errors and provides clear error messages.
    """
    try:
        # Ensure the project path is correct
        repo_root = setup_project_path()
        
        # Dynamically import and run the main program (which handles its own logging setup)
        from src.cds2016.cli import main as run_main
        run_main()
        
    except ImportError as e:
        # Set up minimal logging for error reporting
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Failed to import required modules: {e}")
        logging.error("Please ensure all dependencies are installed (pip install -r requirements.txt)")
        sys.exit(1)

    except KeyboardInterrupt:
        # Set up minimal logging for interrupt reporting
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.info("\nProgram execution interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        # Set up minimal logging for error reporting
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.exception(f"An unexpected error occurred during program execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
