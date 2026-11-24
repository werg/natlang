#!/usr/bin/env python3
"""Serial, owned-device tail of the existing canonical Luna campaign.

The support module supplies the same process/device lease and world metadata used
by actual native qualification. Native references are never submitted to the
teacher pipeline. Only canonical teacher waves enter its archive.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from natlang_neuralese.common.hashing import sha256_file_hex as sha

def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def output(argv):
    return subprocess.check_output(list(map(str, argv)), text=True).strip()


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0


def wait_unit(unit):
    while active(unit) or active(unit.removesuffix('.service') + '-waiter.service'):
        time.sleep(15)
    assert output(['systemctl', '--user', 'show', unit, '-p', 'Result', '--value']) == 'success', unit


def selected_goals(cases, fixtures):
    selected = [item for item in fixtures if item['case']['scenario'] in cases]
    assert len(selected) == 8, 'The V21 training scope has eight Android goals, not all thirteen native goals'
    for item in selected:
        row = cases[item['case']['scenario']]
        assert row['requiresAndroidEmulator'] and row['taskId'] == item['case']['taskId']
        assert hashlib.sha256(json.dumps(item['world']['goal'], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == item['case']['goalSha256']
    # Installed-copy fixtures use the real immutable APK built by the product;
    # bootstrap success is not invented or required as teacher evidence.
    return sorted(selected, key=lambda x: x['world']['id'] != 'android-delivery-visual-first-launch')


def approve_owned_workspace(device, run_json, expected_source, record):
    """The fixture owner approves only its exact, known application via real UI."""
    state = Path(read(run_json)['stateDir'])
    sources = list(state.glob('workspaces/system-*/source/clients/mobile-browser'))
    if len(sources) != 1:
        return
    hashes = {}
    for source in expected_source.rglob('*'):
        if not source.is_file() or 'node_modules' in source.parts or '/dist/' in str(source) or '/build/' in str(source):
            continue
        relative = source.relative_to(expected_source)
        assert sha(sources[0] / relative) == sha(source), 'Never approve unknown workspace code'
        hashes[str(relative)] = sha(source)
    assert len(hashes) >= 180
    xml = device.adb('exec-out', 'uiautomator', 'dump', '/dev/tty').stdout
    start, end = xml.find('<?xml'), xml.rfind('</hierarchy>')
    if start < 0 or end < 0:
        return
    tree = ET.fromstring(xml[start:end + 12])
    labels = [n.attrib.get('text') or n.attrib.get('content-desc') or '' for n in tree.iter('node')]
    if 'Review code before starting' not in labels:
        return
    assert any('https://github.com/panticonic/vibestudio-system.' in label for label in labels)
    assert any('run 1 program on this computer' in label for label in labels)
    parents = {child: parent for parent in tree.iter() for child in parent}
    candidates = [n for n in tree.iter('node') if n.attrib.get('text') == 'Start workspace' or n.attrib.get('content-desc') == 'Start workspace']
    assert candidates
    node = candidates[0]
    while node.attrib.get('clickable') != 'true':
        node = parents[node]
    assert node.attrib.get('enabled') == 'true' and node.attrib.get('package') == 'app.vibestudio.mobile.internal'
    bounds = list(map(int, re.findall(r'\d+', node.attrib['bounds'])))
    assert len(bounds) == 4
    x, y = (bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2
    assert 0 < x < 1080 and 0 < y < 2400
    device.adb('shell', 'input', 'tap', str(x), str(y))
    save(record, {'role': 'fixture-owner-approval', 'modelCalls': 0,
                  'actorScriptSha256': sha(__file__), 'lease': str(device.lease_path),
                  'deviceBootId': device.lease['deviceBootId'],
                  'matchedApplicationSourceFiles': len(hashes),
                  'applicationSourceHashMapSha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                  'uiAction': 'Start workspace', 'at': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--native-support', type=Path, required=True)
    parser.add_argument('--bulk-unit', required=True)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.campaign.resolve()
    candidate = root / 'candidate'
    pins = read(candidate / 'pins.json')
    support = args.native_support.resolve()
    config = read(support / 'inputs.json')
    assert pins['host'] == config['host']['commit']
    assert Path(pins['host_checkout']).resolve() == Path(config['host']['path']).resolve()
    assert Path(pins['template_checkouts']).resolve() == Path(config['selection']).resolve()
    assert Path(pins['release']['manifest']).resolve() == (Path(config['releaseRoot']) / 'userland-builds/release.json').resolve()
    driver = Path(pins['driver_checkout'])
    sys.path[:0] = [str(support), str(driver / 'scripts/neuralese_data'), str(driver / 'training/neuralese'), str(driver)]
    fixture_module = importlib.import_module('android_fixtures')
    from vibestudio_campaign import campaign_scope, attempt_context
    held = args.held
    _, _, cases = campaign_scope(root / 'catalog.jsonl', held)
    metadata = fixture_module.load_metadata()
    goals = selected_goals(cases, metadata['fixtures'])
    host = Path(config['host']['path'])
    apk = host / 'hosts/android/app/build/outputs/apk/internal/app-internal.apk'
    assert apk.is_file() and apk.stat().st_size > 90_000_000
    assert (host / 'dist/main.cjs').is_file()
    assert output(['git', '-C', driver, 'rev-parse', 'HEAD']) == pins['driver']
    assert output(['git', '-C', host, 'rev-parse', 'HEAD']) == config['host']['commit']
    proof = {'schema': 'vibestudio.owned-android-teacher-tail/1',
             'cases': [x['case']['scenario'] for x in goals], 'driver': pins['driver'],
             'sourceHost': config['host']['commit'], 'releaseSha256': sha(Path(config['releaseRoot']) / 'userland-builds/release.json'),
             'nativeSupportSha256': sha(support / 'android_fixtures.py'), 'scriptSha256': sha(__file__),
             'fixtureApk': str(apk), 'fixtureApkSha256': sha(apk), 'fixtureApkBytes': apk.stat().st_size,
             'nativeReferencesUsedAsTeacher': False, 'bulkUnit': args.bulk_unit}
    state = candidate / 'android-tail-v21'
    state.mkdir(exist_ok=True)
    save(state / 'recipe.json', proof)
    if args.prepare_only:
        print(json.dumps(proof))
        return
    wait_unit(args.bulk_unit)
    if (root / 'stop-after-wave').exists():
        raise SystemExit('Campaign owner hold retained; no Android teacher started')
    assert read(candidate / 'pins.json') == pins, 'Release changed while waiting; prepare a successor recipe'
    status = read(root / 'status.json')['cases']
    assert not any(row['status'] == 'pending' and not cases[name].get('requiresAndroidEmulator') for name, row in status.items())
    assert sha(apk) == proof['fixtureApkSha256']
    # A teacher may rebuild the product APK in its normal output directory.
    # Retain these exact prerequisite bytes independently of that mutable path.
    fixture_apk = state / ('fixture-' + proof['fixtureApkSha256'] + '.apk')
    if not fixture_apk.exists():
        shutil.copyfile(apk, fixture_apk)
    assert sha(fixture_apk) == proof['fixtureApkSha256']
    fixture_module.prepare()  # New config only, no ancestor userdata.
    stamp = str(time.time_ns())
    emulator_unit = f'vibe-v21-native-android-owned-teacher-tail-{stamp}.service'
    ledger = ['/usr/bin/python3', str(driver / 'scripts/memory_ledger.py')]
    subprocess.run([*ledger, 'run', '--unit', emulator_unit.removesuffix('.service'), '--budget-gb', '4', '--reserve-gb', '7.5',
                    '--priority', 'low', '--no-resumable', '--hold-budget', '--wait', '604800', '--detach', '--workdir', str(driver),
                    '--env', 'ANDROID_HOME=' + str(fixture_module.SDK), '--env', 'ANDROID_AVD_HOME=' + str(fixture_module.AVD_HOME), '--',
                    str(fixture_module.SDK / 'emulator/emulator'), '-avd', fixture_module.AVD, '-port', '5580', '-no-window', '-no-audio',
                    '-no-boot-anim', '-no-snapshot', '-gpu', 'swiftshader_indirect'], check=True)
    device = None
    try:
        while not active(emulator_unit):
            assert active(emulator_unit.removesuffix('.service') + '-waiter.service')
            time.sleep(15)
        for _ in range(240):
            boot = subprocess.run([str(fixture_module.ADB), '-s', fixture_module.SERIAL, 'shell', 'getprop', 'sys.boot_completed'], capture_output=True, text=True)
            if boot.returncode == 0 and boot.stdout.strip() == '1':
                break
            assert active(emulator_unit)
            time.sleep(5)
        else:
            raise RuntimeError('Owned emulator boot exceeded 20 minutes')
        pid = int(output(['systemctl', '--user', 'show', emulator_unit, '-p', 'MainPID', '--value']))
        proc = Path('/proc') / str(pid)
        lease = {'ownedDisposableEmulator': True, 'avdName': fixture_module.AVD, 'serial': fixture_module.SERIAL,
                 'avdHome': str(fixture_module.AVD_HOME), 'unit': emulator_unit, 'pid': pid,
                 'startTicks': proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19], 'cgroup': proc.joinpath('cgroup').read_text().strip(),
                 'hostBootId': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'deviceBootId': output([fixture_module.ADB, '-s', fixture_module.SERIAL, 'shell', 'cat', '/proc/sys/kernel/random/boot_id'])}
        lease_path = state / ('lease-' + stamp + '.json')
        save(lease_path, lease)
        device = fixture_module.OwnedAndroidFixtures(lease_path, proof['releaseSha256'])
        # The real product-built APK is a prerequisite for installed-copy worlds,
        # never a substitute trajectory or an answer injected into a goal.
        for attempt in range(2):
            for item in goals:
                name = item['case']['scenario']
                if (root / 'stop-after-wave').exists():
                    return
                current = read(root / 'status.json')['cases'][name]
                if current['status'] not in ('pending', 'prerequisite-unavailable'):
                    continue
                device._absent()
                if item['prestate'] != 'absent':
                    assert sha(fixture_apk) == proof['fixtureApkSha256']
                    assert 'Success' in device.adb('install', str(fixture_apk)).stdout
                    assert 'Success' in device.adb('shell', 'pm', 'clear', fixture_module.PACKAGE).stdout
                    device.adb('shell', 'am', 'force-stop', fixture_module.PACKAGE)
                    assert item['prestate'] == 'stopped', 'V21 training has no running-app Android fixture'
                prestate = device.observe()
                assert prestate['installed'] == (item['prestate'] == 'stopped') and not prestate['process']
                epoch, provenance, _ = attempt_context(argparse.Namespace(release_pins=candidate/'pins.json', host=host, attempt_epoch=None), root/'catalog.jsonl')
                with (root / '.controller.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    plans = sorted((root / 'waves/jobs').glob('[0-9]*.json'))
                    assert not any((root/'waves'/f'{p.stem}-capture').exists() and not (root/'waves'/f'{p.stem}-result.json').exists() for p in plans)
                    wave = f'{int(plans[-1].stem) + 1:04d}'
                    job = root / 'waves/jobs' / (wave + '.json')
                    with job.open('x') as target:
                        target.write(json.dumps({'cases': [name], 'attempt_epoch': epoch, 'provenance': provenance}) + '\n')
                save(state / (wave + '-fixture.json'), {'case': name, 'observedPrestate': prestate, 'lease': str(lease_path), 'recipeSha256': sha(state/'recipe.json')})
                unit = f'vibe-semantic-luna-v21-android-{wave}'
                subprocess.run(['bash', str(candidate/'launch-collection.sh'), unit, '1', '10.5', '1', '--max-waves', '1'],
                               env={**os.environ, 'VIBESTUDIO_OWNED_ANDROID_SERIAL': fixture_module.SERIAL}, check=True)
                service = unit + '.service'
                approval = state / (wave + '-fixture-owner-approval.json')
                while active(service) or active(unit + '-waiter.service'):
                    if item['world'].get('route') == 'pair' and item['world'].get('mode') == 'pair' and not approval.exists():
                        for run_json in (root/'waves'/f'{wave}-capture').glob('st_*/run.json'):
                            approve_owned_workspace(device, run_json, Path(config['selection'])/'system/clients/mobile-browser', approval)
                    time.sleep(3)
                assert (root/'waves'/f'{wave}-result.json').is_file(), 'Canonical archive/receipt incomplete; do not continue'
                device._absent()
        final_status = read(root / 'status.json')['cases']
        unfinished = {name: row for name, row in final_status.items()
                      if row['status'] in ('pending', 'prerequisite-unavailable', 'reviewer-required')}
        assert not unfinished, f'Actual unresolved goals remain; preserve evidence instead of claiming exhaustion: {unfinished}'
        if (root / 'stop-after-wave').exists():
            return
        # With no pending goal this performs status/archive finalization only.
        # Keep the device alive until prerequisite accounting is published.
        finalize = f'vibe-semantic-luna-v21-android-finalize-{stamp}'
        subprocess.run(['bash', str(candidate/'launch-collection.sh'), finalize, '1', '10.5', '1', '--max-waves', '1'],
                       env={**os.environ, 'VIBESTUDIO_OWNED_ANDROID_SERIAL': fixture_module.SERIAL}, check=True)
        wait_unit(finalize + '.service')
        assert (root / 'finished.json').is_file()
        save(state / 'completed.json', {'at': time.time(), 'recipeSha256': sha(state/'recipe.json'),
                                        'canonicalFinished': read(root/'finished.json')})
    finally:
        if device is not None:
            device.close()
            assert sha(fixture_apk) == proof['fixtureApkSha256']
            fixture_apk.unlink()
        else:
            # Exact unit was created here; joining it is safe even if boot failed.
            subprocess.run(['systemctl', '--user', 'stop', emulator_unit, emulator_unit.removesuffix('.service')+'-waiter.service'], check=False)


if __name__ == '__main__':
    main()
