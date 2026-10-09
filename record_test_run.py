"""Record an actual suite run, dependency versions, frontend checks and pip check."""
import argparse
from datetime import datetime
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo
from packaging.requirements import Requirement
import config


def environment_packages():
    pending = ['opencv-python', 'numpy', 'pandas', 'scikit-learn', 'flask', 'joblib', 'matplotlib', 'Pillow', 'scipy', 'pytest']
    found = {}
    while pending:
        name = pending.pop()
        distribution = metadata.distribution(name)
        canonical = distribution.metadata['Name'].lower().replace('_', '-')
        if canonical in found: continue
        found[canonical] = distribution.version
        for raw in distribution.requires or []:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({'extra': ''}): pending.append(req.name)
    return dict(sorted(found.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=config.BASE_DIR / 'reports')
    args = parser.parse_args(argv)
    when = datetime.now(ZoneInfo('Asia/Kolkata'))
    stem = f'test_run_py{platform.python_version()}_{when:%Y-%m-%d_%H%M%S}_{uuid4().hex[:8]}'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml = args.output_dir / (stem + '.xml')
    log = args.output_dir / (stem + '.log')
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--junitxml', str(xml)]
    env = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'LECOR_PREVIEW_ONLY': 'false'}
    with log.open('x') as stream:
        test = subprocess.run(command, cwd=config.BASE_DIR, env=env, stdout=stream, stderr=subprocess.STDOUT)
    frontend = subprocess.run(['node', 'tests/frontend_stage1.cjs'], cwd=config.BASE_DIR, capture_output=True, text=True)
    pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    report = {'generated_at': when.isoformat(), 'python': platform.python_version(), 'platform': platform.platform(),
              'machine': platform.machine(), 'packages': environment_packages(), 'pytest_exit': test.returncode,
              'command': command, 'test_environment': {'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'LECOR_PREVIEW_ONLY': 'false'},
              'junit_file': xml.name, 'log_file': log.name,
              'node_version': subprocess.check_output(['node', '--version'], text=True).strip(),
              'frontend_exit': frontend.returncode, 'frontend_output': frontend.stdout + frontend.stderr,
              'pip_check_exit': pip.returncode, 'pip_check_output': pip.stdout + pip.stderr}
    path = args.output_dir / (stem + '.json')
    with path.open('x') as stream: json.dump(report, stream, indent=2)
    print(log.read_text()); print(path)
    return test.returncode or frontend.returncode or pip.returncode


if __name__ == '__main__':
    raise SystemExit(main())
