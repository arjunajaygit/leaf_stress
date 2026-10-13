"""Build Markdown test evidence from recorded runs; does not invent test outcomes."""
import argparse
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET


def render(records, output):
    runs = []
    for path in records:
        path = Path(path); report = json.loads(path.read_text())
        root = ET.parse(path.parent / report['junit_file']).getroot()
        cases = {}
        for case in root.iter('testcase'):
            status = 'FAIL' if case.find('failure') is not None or case.find('error') is not None else ('SKIP' if case.find('skipped') is not None else 'PASS')
            cases[case.get('classname') + '::' + case.get('name')] = status
        runs.append((path, report, cases))
    lines = ['# Actual test results', '', '**No validated classifier exists yet. Classification results cannot be presented as performance.**', '',
             'Generated from actual pytest JUnit records. Tests using synthetic, legacy and development-trained estimators exercise code only; they do not validate physiological or classification claims. Camera, printing and cross-browser behavior: **needs manual verification on device**.', '',
             '| Python | Recorded run start | Pass | Fail | Skip | Pytest exit | Warnings | Frontend | pip check |', '|---|---|---:|---:|---:|---:|---:|---|---|']
    for path, r, cases in runs:
        counts = {s: list(cases.values()).count(s) for s in ('PASS','FAIL','SKIP')}
        warning_counts = re.findall(r'(\d+) warnings?', (path.parent / r['log_file']).read_text())
        warnings = warning_counts[-1] if warning_counts else '0'
        lines.append(f"| {r['python']} | {r['generated_at']} | {counts['PASS']} | {counts['FAIL']} | {counts['SKIP']} | {r['pytest_exit']} | {warnings} | {'PASS' if r['frontend_exit']==0 else 'FAIL'} | {'PASS' if r['pip_check_exit']==0 else 'FAIL'} |")
    lines += ['', '## Environment and raw evidence', '', 'See `environment.md` for exact versions and recreation instructions. Plugin autoload was disabled, bytecode writes disabled and default preview flag false for the suite; individual tests explicitly enable preview. The application does not suppress warnings. Warning counts are read from the actual logs linked below; inspect those logs for warning categories. The per-test table includes NumPy/joblib deserialization checks.', '']
    for path,r,_ in runs:
        lines += [f"- Python {r['python']}; {r['platform']}; Node {r['node_version']}. [Environment record](../reports/{path.name}), [JUnit XML](../reports/{r['junit_file']}), [test log](../reports/{r['log_file']})."]
    lines += ['', '## Per-test outcomes', '', '| Test | '+' | '.join(r['python'] for _,r,_ in runs)+' |', '|---|'+'---|'*len(runs)]
    for name in sorted(set().union(*(set(cases) for _,_,cases in runs))):
        lines.append('| `'+name.replace('|','\\|')+'` | '+' | '.join(cases.get(name,'NOT RUN') for _,_,cases in runs)+' |')
    lines += ['', '## Interpretation', '', 'No executed automated failures in a PASS-only run does not imply scientific readiness. Original single-leaf grouped evaluation remains blocked by missing accepted classes and group provenance. The new whole-plant dataset has all four classes but still lacks verified group/label provenance; its trained classifier is experimental, not validated. The original classifier remains incompatible. Python 3.14 skips the two saved whole-plant deployment tests because that artifact requires its exact Python 3.13 environment. Frontend checks use a minimal DOM, not real browsers. Whole-plant functional timing is documented separately in wholeplant_implementation.md; neither timing nor training-image inference establishes predictive validity.', '']
    Path(output).write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'docs/test_results.md')
    args = parser.parse_args()
    render(args.records, args.output)
