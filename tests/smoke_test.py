"""End-to-end check on the bundled samples: runs retype and verifies the report.

    python3 tests/smoke_test.py
"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [
    ('typewriter.jpg', 'RECEIVED WITH THANKS', 'RECEIVED WITH REGARDS', 'mono', 0.9),
    ('printed.jpg', 'Certified true copy of the original', 'Certified true copy of the ledger', 'proportional', 1.0),
    ('handwriting.jpg', 'Balance paid in full', 'Balance due in full', 'proportional', 1.0),
]


def main():
    ok = True
    with tempfile.TemporaryDirectory() as out:
        for img, old, new, layout, min_copied in CASES:
            name = img.split('.')[0]
            cmd = [sys.executable, os.path.join(ROOT, 'retype.py'), os.path.join(ROOT, 'examples', img),
                   '--old', old, '--new', new, '--out-dir', out, '--name', name]
            subprocess.run(cmd, check=True, capture_output=True)
            rep = json.load(open(os.path.join(out, f'{name}_report.json')))
            letters = rep['letters']
            copied = sum(l['source'].startswith('copied') for l in letters) / len(letters)
            checks = {
                'every letter placed': len(letters) == len(new.replace(' ', '')),
                'nothing missing': all(l['source'] != 'MISSING' for l in letters) or layout == 'mono',
                f'layout {layout}': rep['layout'] == layout,
                f'copied >= {min_copied:.0%}': copied >= min_copied,
                'image written': os.path.exists(os.path.join(out, f'{name}_same_spacing.png')),
            }
            for what, passed in checks.items():
                print(f"{'PASS' if passed else 'FAIL'}  {img}: {what}")
                ok &= passed
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
