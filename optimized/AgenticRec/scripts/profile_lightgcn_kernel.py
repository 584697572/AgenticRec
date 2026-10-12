"""Train-only numerical parity and isolated kernel timing; no holdout loading."""
import argparse
import json
from pathlib import Path
import random
import statistics
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
    _, nu, ni, seen, edges = load_train_contract()
    graph = normalized_bipartite_graph(nu, ni, edges)
    users = torch.tensor([u for u, _ in edges[:2048]])
    positives = torch.tensor([i for _, i in edges[:2048]])
    sampler = TrainNegativeSampler(ni, seen, random.Random(42))
    negatives = torch.tensor([sampler.sample(int(u)) for u in users])
    records = []
    for threads in (1, 2):
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        original = LightGCN(nu, ni, 64, 1, graph)
        optimized = ResidualLightGCN(nu, ni, 64, 1, graph, [0.5, 0.5])
        optimized.load_state_dict(original.state_dict())
        reference = None
        for name, model in [('original_coo', original), ('cached_csr', optimized)]:
            seconds = []
            for _ in range(6):
                model.zero_grad()
                tick = time.perf_counter()
                loss = lightgcn_loss(model, users, positives, negatives, 0.0001)
                loss.backward()
                seconds.append(time.perf_counter()-tick)
            gradients = [p.grad.clone() for p in model.parameters()]
            if reference is None:
                reference = (float(loss), gradients)
            errors = [float((a-b).abs().max()) for a, b in zip(gradients, reference[1])]
            if max(errors) > 1e-6 or abs(float(loss)-reference[0]) > 1e-7:
                raise ValueError('Real graph forward/backward parity failed')
            records.append({'threads': threads, 'backend': name, 'seconds': seconds,
                            'warm_median_seconds': statistics.median(seconds[1:]),
                            'loss': float(loss), 'loss_difference': float(loss)-reference[0],
                            'gradient_max_abs': max(errors)})
    report = {'status': 'PASS', 'source_sha256': digest(Path(__file__)),
              'model_source_sha256': digest(ROOT/'AgenticRec/src/agenticrec/models/lightgcn_residual.py'),
              'train_edges': len(edges), 'nodes': nu+ni, 'batch_size': 2048, 'dim': 64,
              'results': records, 'torch': str(torch.__version__),
              'scope': 'Isolated same-minibatch forward/backward, excludes optimizer/evaluation; not online speed',
              'test_access': False, 'remote_api_requests': 0}
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
