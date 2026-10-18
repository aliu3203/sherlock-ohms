"""Split 70/20/10 by build, never by individual photo.

Default naming: BUILD__PHOTO.png (also accepts BUILD_PHOTO.png). A custom regex
with a named 'build' group supports build IDs containing underscores.
Ratios apply to build counts; image counts depend on photos per build.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import shutil

DEFAULT_PATTERN = r'^(?P<build>.+?)(?:__|_)'
SPLITS = ('train', 'val', 'test')


def build_id(path, pattern=DEFAULT_PATTERN):
    match = re.match(pattern, Path(path).stem)
    if match is None or not match.groupdict().get('build'):
        raise ValueError(f'No build prefix in {path}; use BUILD__PHOTO or --build-pattern')
    return match.group('build')


def split_by_build(images, *, seed=42, ratios=(.7, .2, .1), pattern=DEFAULT_PATTERN):
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios)-1) > 1e-8:
        raise ValueError('Three positive split ratios must sum to 1')
    groups = defaultdict(list)
    for image in sorted(map(Path, images)):
        groups[build_id(image, pattern)].append(image)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    counts = [int(len(keys)*ratio) for ratio in ratios]
    remainders = sorted(range(3), key=lambda i: (len(keys)*ratios[i]-counts[i], -i), reverse=True)
    for i in remainders[:len(keys)-sum(counts)]:
        counts[i] += 1
    # With at least three builds every split should be usable.
    if len(keys) >= 3:
        for i in range(3):
            if counts[i] == 0:
                donor = max(range(3), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] = 1
    result, offset = {}, 0
    for name, count in zip(SPLITS, counts):
        selected = keys[offset:offset+count]
        result[name] = [image for key in selected for image in groups[key]]
        offset += count
    return result


def split_dataset(images_dir, labels_dir, output_dir, *, seed=42, pattern=DEFAULT_PATTERN):
    images_dir, labels_dir, output_dir = map(Path, (images_dir, labels_dir, output_dir))
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise NotADirectoryError('Both image and label directories must exist')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output must be empty; refusing to mix dataset splits')
    sources = sorted(p for p in images_dir.rglob('*')
                     if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not sources:
        raise ValueError('No source images')
    # Validate all labels/collisions before copying anything.
    targets = set()
    for source in sources:
        relative = source.relative_to(images_dir)
        label = labels_dir / relative.with_suffix('.txt')
        if not label.is_file():
            raise FileNotFoundError(f'Missing label: {label}')
        key = relative.with_suffix('.txt')
        if key in targets:
            raise ValueError(f'Two images would share a label: {key}')
        targets.add(key)
    splits = split_by_build(sources, seed=seed, pattern=pattern)
    manifest = {}
    for name, paths in splits.items():
        (output_dir / 'images' / name).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / name).mkdir(parents=True, exist_ok=True)
        manifest[name] = []
        for source in paths:
            relative = source.relative_to(images_dir)
            label_relative = relative.with_suffix('.txt')
            target = output_dir / 'images' / name / relative
            label_target = output_dir / 'labels' / name / label_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            shutil.copy2(labels_dir / label_relative, label_target)
            manifest[name].append({'image': relative.as_posix(),
                                   'build': build_id(source, pattern)})
    (output_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--labels-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--build-pattern', dest='pattern', default=DEFAULT_PATTERN)
    result = split_dataset(**vars(parser.parse_args()))
    print({name: len(paths) for name, paths in result.items()})


if __name__ == '__main__':
    main()
