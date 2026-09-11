"""Verify a released asset bundle, or remap its Blender dependencies in memory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(root):
    return json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes asset bundle: {relative}')
    return path


def verify(root):
    root = Path(root)
    manifest = read_manifest(root)
    for item in manifest['files']:
        path = contained(root, item['path'])
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Asset checksum mismatch: {path}')
    for case in manifest['cases']:
        meta = json.loads(contained(root, case['simulation_assets'] + '/assets.json').read_text(encoding='utf-8'))
        if sha256(contained(root, case['blend'])) != meta['source_blend_sha256']:
            raise ValueError(f'Appearance/collision mismatch: {case["id"]}')
        if sha256(contained(root, case['simulation_assets'] + '/geometry_and_fill.npz')) != meta['geometry_sha256']:
            raise ValueError(f'Collision/fill mismatch: {case["id"]}')
    return manifest


def remap_loaded_images(root):
    """Call AFTER bpy.ops.wm.open_mainfile. Does not save or modify originals."""
    import bpy
    root = Path(root)
    manifest = read_manifest(root)
    current = sha256(Path(bpy.data.filepath))
    known = {item['sha256'] for item in manifest['files'] if item['path'].endswith('.blend')}
    if current not in known:
        raise ValueError('Loaded blend is not an original from this bundle')
    if bpy.data.libraries:
        raise ValueError('Unexpected linked libraries; a new dependency audit is required')
    mappings = {item['stored_path'].replace('\\', '/'): item for item in manifest['dependencies']}
    loaded = []
    for image in bpy.data.images:
        if image.packed_file or image.source in ('VIEWER', 'GENERATED'):
            continue
        if image.source != 'FILE' or not image.filepath:
            raise ValueError(f'Unsupported external image: {image.name}')
        stored = image.filepath.replace('\\', '/')
        entry = mappings.get(stored)
        if entry is None:
            # Also permit this function to be called twice in the same session.
            entry = next((x for x in mappings.values() if str(contained(root, x['path'])) == stored), None)
        if entry is None:
            raise ValueError(f'Unmapped image dependency: {stored}')
        path = contained(root, entry['path'])
        if sha256(path) != entry['sha256']:
            raise ValueError(f'External image checksum mismatch: {path}')
        image.filepath = str(path)
        image.reload()
        if min(image.size) <= 0 or not image.has_data:
            raise ValueError(f'Image could not be loaded: {path}')
        loaded.append(dict(name=image.name, path=str(path), size=list(image.size)))
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--blender-check', action='store_true')
    parser.add_argument('--report', type=Path)
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    manifest = verify(args.root)
    report = dict(verified=True, release=manifest['release'], files=len(manifest['files']),
                  total_bytes=sum(f['bytes'] for f in manifest['files']), rendered=False,
                  simulation_started=False, cases=[])
    if args.blender_check:
        import bpy
        import numpy as np
        report['blender_version'] = bpy.app.version_string
        for case in manifest['cases']:
            blend = contained(args.root, case['blend'])
            bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
            images = remap_loaded_images(args.root)
            for name in (case['moving_object'], 'DesignView_00'):
                if name not in bpy.data.objects:
                    raise ValueError(f'Missing scene object: {name}')
            with np.load(contained(args.root, case['simulation_assets'] + '/geometry_and_fill.npz'), allow_pickle=False) as arrays:
                count = len(arrays['positions'])
                if count != case['particle_count']:
                    raise ValueError('Unexpected particle count')
            report['cases'].append(dict(id=case['id'], particle_count=count, images=images,
                                        camera_present=True, moving_object_present=True))
        verify(args.root)  # Inputs remain unchanged after in-memory path remapping.
    if args.report:
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
