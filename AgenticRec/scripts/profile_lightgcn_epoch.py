"""Matched warm one-epoch training timing and parameter parity, train only."""
import argparse
import json
from pathlib import Path
import random
import time

import torch

from agenticrec.data import digest
from agenticrec.models.bpr import TrainNegativeSampler
from agenticrec.models.lightgcn import LightGCN, lightgcn_loss, normalized_bipartite_graph
from agenticrec.models.lightgcn_residual import ResidualLightGCN
from agenticrec.training import ROOT, load_train_contract


def run(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Output must be new and inside project')
    torch.set_num_threads(2)
    manifest, nu, ni, seen, edges = load_train_contract()
    graph = normalized_bipartite_graph(nu, ni, edges)
    torch.manual_seed(42)
    original = LightGCN(nu, ni, 64, 1, graph)
    optimized = ResidualLightGCN(nu, ni, 64, 1, graph, [0.5, 0.5])
    optimized.load_state_dict(original.state_dict())
    results = []
    reference, losses_reference = None, None
    for name, model in [('original_coo', original), ('cached_csr_sum', optimized)]:
        pairs = list(edges)
        rng = random.Random(42)
        rng.shuffle(pairs)
        sampler = TrainNegativeSampler(ni, seen, rng)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = []
        tick = time.perf_counter()
        for offset in range(0, len(pairs), 2048):
            batch = pairs[offset:offset+2048]
            users = torch.tensor([u for u, _ in batch])
            pos = torch.tensor([i for _, i in batch])
            neg = torch.tensor([sampler.sample(u) for u, _ in batch])
            optimizer.zero_grad()
            loss = lightgcn_loss(model, users, pos, neg, 0.0001)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        elapsed = time.perf_counter()-tick
        parameters = [p.detach().clone() for p in model.parameters()]
        if reference is None:
            reference, losses_reference = parameters, losses
        errors = [float((a-b).abs().max()) for a, b in zip(parameters, reference)]
        max_loss_difference = max(abs(a-b) for a, b in zip(losses, losses_reference))
        # Repeated float32 optimizer updates amplify tiny parallel reductions.
        if max(errors) > 1e-5 or max_loss_difference > 1e-6:
            raise ValueError('One-epoch trajectory outside preregistered numerical tolerance')
        results.append({'backend': name, 'elapsed_seconds': elapsed, 'steps': len(losses),
                        'train_triples': len(pairs), 'mean_batch_loss': sum(losses)/len(losses),
                        'parameter_max_abs': max(errors), 'max_batch_loss_difference': max_loss_difference})
        print(json.dumps(results[-1]), flush=True)
    output.write_text(json.dumps({'status': 'PASS', 'results': results,
        'source_sha256': digest(Path(__file__)),
        'model_source_sha256': digest(ROOT/'AgenticRec/src/agenticrec/models/lightgcn_residual.py'),
        'train_positive_sha256': manifest['files']['train_positive.parquet']['sha256'],
        'scope': 'Serialized warm one-epoch loop including sampling/tensors/loss/backward/optimizer; excludes data/model setup, validation and cold start',
        'replicates': 1, 'order': ['original_coo', 'cached_csr_sum'],
        'limit': 'Single local paired timing, not entire training or online latency',
        'test_access': False, 'remote_api_requests': 0}, sort_keys=True, indent=2)+'\n',
        encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
