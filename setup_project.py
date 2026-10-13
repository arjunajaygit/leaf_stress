"""Install/check the current Windows classifier + CPU SAM deployment after cloning."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def deployment_files():
    deployment = json.loads((ROOT/'wholeplant_deployment_v1.json').read_text(encoding='utf-8'))
    for key in ('model', 'report'):
        path = (ROOT/deployment[key]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Missing repository deployment {key}: {deployment[key]}. Include the selected artifact folder in the Git commit.')
    return deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate setup without installing or downloading anything')
    args = parser.parse_args()
    deployment = deployment_files()
    report = json.loads((ROOT/deployment['report']).read_text(encoding='utf-8'))
    required = report['library_versions']['python']
    if platform.python_version() != required:
        raise SystemExit(f'This saved classifier needs CPython {required}; you are running {platform.python_version()}. Install the exact version, then recreate the virtual environment. Do not bypass the model contract.')
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Run with the project virtual environment: .\\.venv-py313\\Scripts\\python.exe setup_project.py')
    if not args.check_only:
        if sys.platform != 'win32' or platform.machine().lower() not in ('amd64', 'x86_64'):
            raise SystemExit('The combined lock targets Windows x64. See docs/team_setup.md for other platforms.')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT/'requirements-windows-sam.lock.txt')], check=True, cwd=ROOT)
        subprocess.run([sys.executable, str(ROOT/'setup_sam.py')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, cwd=ROOT)
    os.environ['LECOR_ANALYSIS_MODE'] = 'wholeplant'
    os.environ['LECOR_MODEL_PATH'] = str(ROOT/deployment['model'])
    os.environ['LECOR_BENCHMARK_PATH'] = str(ROOT/deployment['report'])
    os.environ['LECOR_PREVIEW_ONLY'] = 'false'
    import app
    if app.primary_model is None:
        raise SystemExit(app.model_load_error)
    import sam_segmentation
    readiness = sam_segmentation.status()
    if not readiness['available']:
        raise SystemExit(readiness['message'])
    import torch
    import torchvision
    from setup_sam import verify_checkpoint
    verify_checkpoint(sam_segmentation.CHECKPOINT)
    print('Classifier:', deployment['model'])
    print('SAM ready:', torch.__version__, '/', torchvision.__version__, '/', sam_segmentation.DEVICE)
    print('Setup verified. Run .\\.venv-py313\\Scripts\\python.exe run_wholeplant.py')


if __name__ == '__main__':
    main()
