"""CPU-only dependency inspection; never render or save the input blend files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import bpy


def local_path(raw, blend):
    normalized = raw.replace('\\', '/')
    if re.match(r'^[A-Za-z]:/', normalized):
        return Path('/mnt') / normalized[0].lower() / normalized[3:]
    if normalized.startswith('//'):
        return (blend.parent / normalized[2:]).resolve()
    return Path(normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    results = []
    hashes = {}
    for blend in args.blend:
        blend = blend.resolve()
        bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
        paths = sorted(set(bpy.utils.blend_paths(absolute=False, packed=False, local=False)))
        dependencies = []
        for raw in paths:
            if not raw or raw == '<builtin>':
                continue
            path = local_path(raw, blend)
            record = dict(stored_path=raw, local_path=str(path), exists=path.is_file())
            if path.is_file():
                if str(path) not in hashes:
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(chunk)
                    hashes[str(path)] = digest.hexdigest()
                record.update(bytes=path.stat().st_size, sha256=hashes[str(path)])
            dependencies.append(record)
        results.append(dict(blend=str(blend), dependencies=dependencies,
            libraries=[dict(name=item.name, filepath=item.filepath) for item in bpy.data.libraries],
            images=[dict(name=item.name, filepath=item.filepath, source=item.source,
                         packed=bool(item.packed_file), users=item.users) for item in bpy.data.images],
            object_count=len(bpy.data.objects), cameras=[o.name for o in bpy.data.objects if o.type == 'CAMERA']))
    report = dict(blender_version=bpy.app.version_string, inspection_only=True,
                  input_files_saved=False, rendered=False, scenes=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
