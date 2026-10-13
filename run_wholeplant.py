"""Launch the separate experimental whole-plant deployment; retain legacy defaults."""
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    deployment = json.loads((root / 'wholeplant_deployment_v1.json').read_text())
    os.environ['LECOR_ANALYSIS_MODE'] = 'wholeplant'
    os.environ.setdefault('LECOR_MODEL_PATH', deployment['model'])
    os.environ.setdefault('LECOR_BENCHMARK_PATH', deployment['report'])
    from app import main as serve
    serve()


if __name__ == '__main__':
    main()
