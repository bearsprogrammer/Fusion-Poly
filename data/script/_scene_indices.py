"""Export scene-name suffixes, not positions in the scene.json array."""
import argparse
import json
import re
import sys
from pathlib import Path

from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = '/home/allen-5070ti/dataset/nuScenes/full_trainval'


def get_scene_indices(dataset_root, split):
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    scene_path = Path(dataset_root).expanduser() / 'v1.0-trainval' / 'scene.json'
    with scene_path.open() as stream:
        scenes = json.load(stream)
    names = [scene['name'] for scene in scenes]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate scene names in ' + str(scene_path))
    official = create_splits_scenes()[split]
    missing = sorted(set(official) - set(names))
    if missing:
        raise ValueError('Dataset is missing {} {} scenes: {}'.format(len(missing), split, ', '.join(missing)))
    if any(re.fullmatch(r'scene-\d{4}', name) is None for name in official):
        raise ValueError('Unexpected scene name format in official split')
    indices = sorted(int(name.split('-')[1]) for name in official)
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate indices in official split')
    return indices


def main(split):
    parser = argparse.ArgumentParser(
        description='Print/save official nuScenes {} scene numbers: scene-0003 -> 3. '
                    'These are NOT scene.json list positions.'.format(split))
    parser.add_argument('--nusc_path', default=DEFAULT_DATASET, help='Dataset root containing v1.0-trainval/')
    parser.add_argument('--output_path', default=str(ROOT / 'data' / (split + '_indice.json')))
    args = parser.parse_args()
    indices = get_scene_indices(args.nusc_path, split)
    output = Path(args.output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(indices, indent=2) + '\n'
    output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')
    print('Saved {} {} scene indices to {}'.format(len(indices), split, output.resolve()), file=sys.stderr)
