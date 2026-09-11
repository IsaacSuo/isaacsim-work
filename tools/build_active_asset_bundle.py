"""Assemble selected unchanged active-drive inputs; no simulation or rendering."""
import argparse
import json
from pathlib import Path
import shutil
from active_asset_bundle import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/coupled_scenes'
RELEASE = 'active_drive_20260912_v1'
CASES = [
    ('01_container_transfer', 'active_pour_lowered_design_v7', 'active_pour_water_minus10_assets_v10', 'PouringPitcher', 'Accepted lowered pitcher, v10 fill; embedded case metadata is historical, not replacement geometry.'),
    ('02_stirring', 'active_drive_contact_v4/02_stirring', 'active_02_stirring_assets_v2', 'QuietThreeBladeImpeller', 'Seam-repaired v4 appearance. Two-times speed is a RUN parameter (3.6 rad/s), not an asset edit.'),
    ('03_piston_push', 'active_drive_contact_v4/03_piston_push', 'active_03_piston_push_assets_v2', 'ConcealedCarriagePiston', 'Seam-repaired v4 appearance with 4mm fill.'),
    ('03_piston_push_3mm', 'active_drive_contact_v4/03_piston_push', 'active_03_piston_push_assets_v3_3mm', 'ConcealedCarriagePiston', 'Optional 3mm fill from the stationary diagnostic; NOT the default stirring input.'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    if len(audit['scenes']) != 3 or any(s['libraries'] for s in audit['scenes']):
        raise ValueError('Expected three audited, self-contained designs without linked libraries')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    cases = []

    def copy(source, relative):
        target = args.output / relative
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError('Duplicate bundle path with different content')
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha256(target)
        if digest != sha256(source):
            raise ValueError('Input changed while copying')
        records.append(dict(path=relative, bytes=target.stat().st_size, sha256=digest))

    for key, design, assets, moving, note in CASES:
        meta = json.loads((OUTPUT / assets / 'assets.json').read_text(encoding='utf-8'))
        original_id = meta['case']['id']
        blend = OUTPUT / design / (original_id + '.blend')
        geometry = OUTPUT / assets / 'geometry_and_fill.npz'
        if sha256(blend) != meta['source_blend_sha256'] or sha256(geometry) != meta['geometry_sha256']:
            raise ValueError(f'Source pair mismatch: {key}')
        if str(blend.resolve()) not in {s['blend'] for s in audit['scenes']}:
            raise ValueError('Design was not audited')
        design_rel = 'designs/' + original_id
        sim_rel = 'simulation_assets/' + key
        for name in (original_id + '.blend', 'design_spec.json', 'cameras.json'):
            copy(OUTPUT / design / name, design_rel + '/' + name)
        for name in ('assets.json', 'geometry_and_fill.npz'):
            copy(OUTPUT / assets / name, sim_rel + '/' + name)
        cases.append(dict(id=key, blend=design_rel + '/' + original_id + '.blend',
            simulation_assets=sim_rel, moving_object=moving, particle_spacing_m=meta['particle_spacing_m'],
            particle_count=meta['particle_count'], source_asset_directory=assets, note=note))
    copy(args.audit, 'dependency_audit_local.json')
    copy(ROOT / 'tools/active_asset_bundle.py', 'support/active_asset_bundle.py')
    copy(ROOT / 'tools/ACTIVE_ASSET_BUNDLE_README.md', 'README.md')
    dependencies = {}
    for scene in audit['scenes']:
        for dep in scene['dependencies']:
            if not dep['exists']:
                raise ValueError('Unresolved dependency')
            if dep['stored_path'] != 'Y:/scenes/HDRI/bryanston_park_sunrise_8k.exr':
                raise ValueError('Unexpected dependency; explicit transfer plan required')
            relative = 'dependencies/HDRI/bryanston_park_sunrise_8k.exr'
            dependencies[dep['stored_path']] = dict(stored_path=dep['stored_path'], path=relative,
                bytes=dep['bytes'], sha256=dep['sha256'],
                server_reuse_source='/data/jiachen/isaacsim_work/scenes/HDRI/bryanston_park_sunrise_8k.exr')
    # The verified identical HDRI will be copied locally on the server, not uploaded.
    network_bytes = sum(r['bytes'] for r in records)
    records.extend(dict(path=d['path'], bytes=d['bytes'], sha256=d['sha256']) for d in dependencies.values())
    manifest = dict(schema_version=1, release=RELEASE, immutable_originals=True,
        related_request='20260912_001_stirring_2x_handoff',
        scope='Appearance, matching collision/fill and external images only; application code and simulation results are NOT included.',
        local_payload_bytes=network_bytes, files=records, cases=cases, dependencies=list(dependencies.values()))
    with (args.output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(release=RELEASE, cases=len(cases), network_bytes=network_bytes,
                         final_bytes=sum(f['bytes'] for f in records)), indent=2))


if __name__ == '__main__':
    main()
