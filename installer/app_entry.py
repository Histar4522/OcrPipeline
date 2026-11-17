"""Cross-platform PyInstaller entry point for the OcrPipeline desktop app."""

import multiprocessing
import sys

from ocr_pipeline.desktop import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
