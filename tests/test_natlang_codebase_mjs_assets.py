from pathlib import Path

from scripts.rl_tasks.intake_natlang_codebase import _project


def test_intake_includes_text_service_adapters(tmp_path: Path):
    package = tmp_path / 'codebase' / 'package'
    package.mkdir(parents=True)
    (package / 'natlang.json').write_text('{}\n')
    (package / 'main.nl').write_text('---\nargs: {}\nreturns: string\n---\nRead.\n')
    (package / 'service.mjs').write_text('export const service = { read: async () => 1 };\n')
    (package / 'NOTICE.md').write_text('Attribution\n')
    files = _project(package, {'package_assets': ['NOTICE.md']})
    assert files['service.mjs'] == 'export const service = { read: async () => 1 };\n'
    assert files['NOTICE.md'] == 'Attribution\n'
    assert 'main.nl' in files
