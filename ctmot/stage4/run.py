"""Explicit stage ordering: local cases -> all-track small run -> full val -> official eval."""
import argparse
import sys
from pathlib import Path
from .common import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='ctmot/stage4/config.yaml')
    parser.add_argument('--stage',choices=['all','multicase','small','full','evaluate','report'],default='all')
    args=parser.parse_args();cfg=load_config(args.config);root=output_root(cfg)
    cases=sorted({c['scene'] for c in cfg['cases']})
    if args.stage in ('all','multicase'):
        from .multicase import run
        prepare(cfg,cases);run(cfg)
    if args.stage in ('all','small'):
        if not read_json(root/'multicase/report.json')['stage3_reproduced']:
            raise ValueError('Complete multicase first')
        from .tracking import run
        prepare(cfg,cases);run(cfg,cases,label='small')
    if args.stage in ('all','full'):
        from nuscenes.utils.splits import create_splits_scenes
        small=read_json(root/'tracking_small.json')
        if small['b0_ids_exact_numeric_tolerance']!=1e-8:
            raise ValueError('Small B0 reproduction must pass first')
        from .tracking import run
        provenance=dict(command=' '.join(sys.argv),python=sys.executable,
            input_hashes=input_provenance(cfg),config=cfg,
            code_hashes={str(p):sha256(p) for d in ('ctmot','tracking','motion_module','dataloader','pre_processing','geometry','utils') for p in resolve(d).rglob('*.py')})
        write_json(root/'run_manifest.json',provenance)
        (root/'commands.txt').write_text('export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fusionpoly-mpl\n'+
            sys.executable+' -m ctmot.stage4.run --config '+args.config+' --stage all\n'+
            '\n'.join(sys.executable+' -m ctmot.stage4.evaluation --variant '+v for v in ['B0','B1','B2','B3'])+'\n')
        names=sorted(create_splits_scenes()['val']);assert len(names)==150
        prepare(cfg,names);run(cfg,names,label='full')
        if input_provenance(cfg)!=provenance['input_hashes']: raise AssertionError('Original inputs changed')
    if args.stage in ('all','full','evaluate'):
        from .evaluation import evaluate_variant
        for variant in ['B0','B1','B2','B3']: evaluate_variant(cfg,variant)
    if args.stage in ('all','full','evaluate','report'):
        from .report import generate
        generate(cfg)


if __name__=='__main__': main()
