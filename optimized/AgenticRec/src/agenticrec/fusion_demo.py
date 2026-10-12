"""Offline, opt-in inference; stable CLI defaults and paid API budgets are untouched."""
import argparse
import json
from pathlib import Path

import torch

from .adapters.model import KnownUserScorer
from .adapters.validated import load_validated_scorer
from .fusion_pipeline import FusionPipeline
from .pipeline import FixedRequest
from .retrieval.adaptive import CachedResearchRetriever
from .training import ROOT


def write_diagnostics(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('Candidate traces must stay in local ignored artifacts')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')


def build_pipeline(kind, seed, *, profile, mode):
    if profile not in ('baseline', 'validated-20261011'):
        raise ValueError('Unknown model profile')
    known = (load_validated_scorer(kind, seed) if profile == 'validated-20261011'
             else KnownUserScorer.from_artifacts(kind, seed))
    return FusionPipeline(CachedResearchRetriever.from_frozen(known), mode=mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--model', choices=('bpr', 'lightgcn'), required=True)
    parser.add_argument('--seed', type=int, choices=(7, 42, 2026), default=42)
    parser.add_argument('--profile', choices=('baseline', 'validated-20261011'), required=True)
    parser.add_argument('--mode', choices=('rrf', 'no_content', 'fixed', 'conditional'), required=True)
    parser.add_argument('--diagnostics', type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    try:
        request = FixedRequest.from_json_file(args.request)
        pipeline = build_pipeline(args.model, args.seed, profile=args.profile, mode=args.mode)
        response = pipeline.recommend(request).to_dict()
        if args.diagnostics:
            write_diagnostics(args.diagnostics, pipeline.diagnose(request))
        response['model_profile'] = args.profile
        response['fusion_mode'] = args.mode
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({'status': 'PIPELINE_ERROR', 'error_type': type(error).__name__,
                          'reason': str(error), 'remote_api_requests': 0}))
        return 3
    print(json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if response['status'] == 'OK' else 2


if __name__ == '__main__':
    raise SystemExit(main())
