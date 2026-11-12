from pathlib import Path
import csv,hashlib
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'MANIFEST_SHA256.csv').open(encoding='utf-8')))
expected={r['file'] for r in rows}
for r in rows:
    p=(root/r['file']).resolve()
    assert root in p.parents, 'Manifest path outside package'
    assert p.is_file(),r['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['file']
actual={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='MANIFEST_SHA256.csv' and p.relative_to(root).parts[0] not in {'outputs','data','.venv'}}
assert actual==expected,(actual-expected,expected-actual)
print(f'Package integrity verified: {len(rows)} files')

