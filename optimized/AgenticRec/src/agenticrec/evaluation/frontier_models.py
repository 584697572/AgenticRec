"""Small, actually trained method probes; not industrial-paper reproductions."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import pyarrow.parquet as pq
import torch
from torch import nn

from .comparison_cases import ROOT, write_new, sha
from .frontier_state import GENRES, empty_state, merge_state, policy_features


class StatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(10, 24), nn.Tanh(), nn.Linear(24, 3))

    def forward(self, x):
        return self.network(x)

    def action(self, previous, delta, message):
        with torch.no_grad():
            x = torch.tensor(policy_features(previous, delta, message), dtype=torch.float32)
            return int(self(x).argmax())


def policy_training_data(seed, n=900):
    rng = random.Random(seed)
    rows, rewards = [], []
    for _ in range(n):
        previous = empty_state()
        previous.update(include_genres=[rng.choice(['Drama', 'Action', 'Comedy'])],
                        year_min=1980, year_max=2000)
        delta = dict(include_genres=[rng.choice(['Thriller', 'Romance', 'Crime'])],
                     exclude_genres=None, year_min=None, year_max=None, k=10,
                     liked_item_ids=[], seen_item_ids=[], exclude_previous=False,
                     reset_constraints=False, genre_mode='replace', evidence={})
        kind = rng.randrange(3)
        if kind == 1:
            message = 'Discard all previous conditions. Recommend 10 movies.'
            correct = dict(delta, reset_constraints=True)
            delta['reset_constraints'] = rng.random() > .6
        elif kind == 2:
            message = 'Keep previous conditions and also require another genre.'
            correct = dict(delta, genre_mode='add')
            delta['genre_mode'] = 'add' if rng.random() > .6 else 'replace'
        else:
            message = 'Switch to a different genre instead. Keep previous year limits.'
            correct = dict(delta)
        expected = merge_state(previous, correct)
        actual = [merge_state(previous, delta, a) for a in range(3)]
        # Binary full-state success is the environment reward. All equivalent
        # successful actions receive the same reward, rather than guessed loss.
        rows.append(policy_features(previous, delta, message))
        rewards.append([float(state == expected) for state in actual])
    return torch.tensor(rows), torch.tensor(rewards)


def train_policy(directory, seed):
    torch.manual_seed(seed)
    x, rewards = policy_training_data(seed)
    model = StatePolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    losses = []
    target = rewards.argmax(1)
    for step in range(120):
        loss = nn.functional.cross_entropy(model(x), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 20 == 0:
            losses.append({'stage': 'sft', 'step': step, 'loss': float(loss.detach())})
    sft = directory / f'policy_sft_{seed}.pt'
    torch.save(model.state_dict(), sft)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    # Actual sampled REINFORCE with a detached per-state expected-reward baseline.
    for step in range(100):
        logits = model(x)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        reward = rewards.gather(1, actions[:, None]).squeeze(1)
        baseline = (distribution.probs.detach() * rewards).sum(1)
        loss = -(distribution.log_prob(actions) * (reward-baseline)).mean() - .003 * distribution.entropy().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 20 == 0:
            losses.append({'stage': 'reinforce', 'step': step, 'loss': float(loss.detach()),
                           'sampled_reward': float(reward.mean())})
    path = directory / f'policy_rl_{seed}.pt'
    torch.save(model.state_dict(), path)
    audit_x, audit_r = policy_training_data(seed+10000, 300)
    with torch.no_grad():
        accuracy = audit_r.gather(1, model(audit_x).argmax(1)[:, None]).mean().item()
    return {'seed': seed, 'training_states': len(x), 'sft_steps': 120, 'rl_steps': 100,
            'algorithm': 'sampled REINFORCE on binary full-state correctness, not Rank-GRPO',
            'audit_generated_state_success': accuracy, 'curves': losses,
            'sft_checkpoint_sha256': sha(sft), 'rl_checkpoint_sha256': sha(path),
            'parameter_l2_change': math.sqrt(sum(float((v-initial[k]).square().sum())
                                                for k, v in model.state_dict().items()))}


class CatalogGenerator(nn.Module):
    """Catalog-ID autoregression, trainable semantics and sequential context."""
    def __init__(self, n_items, metadata, dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_items+1, dim, padding_idx=0)
        self.semantic = nn.Linear(len(GENRES)+1, dim, bias=False)
        self.encoder = nn.GRU(dim, dim, batch_first=True)
        self.decoder = nn.GRUCell(dim, dim)
        self.intent = nn.Linear(len(GENRES)+2, dim)
        self.bias = nn.Parameter(torch.zeros(n_items+1))
        self.register_buffer('metadata', metadata)

    def item_vectors(self):
        return self.embedding.weight + self.semantic(self.metadata)

    def encode(self, history, intent):
        vectors = self.item_vectors()
        _, hidden = self.encoder(vectors[history])
        return hidden.squeeze(0) + torch.tanh(self.intent(intent))

    def logits(self, hidden):
        return hidden @ self.item_vectors().T / math.sqrt(hidden.shape[-1]) + self.bias

    def step(self, hidden, item_ids):
        return self.decoder(self.item_vectors()[item_ids], hidden)

    def generate(self, history, intent, eligible, k):
        if not eligible:
            return []
        with torch.no_grad():
            hidden = self.encode(torch.tensor([history[-5:] or [0]]), torch.tensor([intent]))
            allowed = torch.zeros(self.embedding.num_embeddings, dtype=torch.bool)
            allowed[eligible] = True
            result = []
            for _ in range(min(k, len(eligible))):
                scores = self.logits(hidden).squeeze(0).masked_fill(~allowed, float('-inf'))
                chosen = int(scores.argmax())
                result.append(chosen); allowed[chosen] = False
                hidden = self.step(hidden, torch.tensor([chosen]))
            return result


def metadata_tensor(n_items, catalog):
    metadata = torch.zeros(n_items+1, len(GENRES)+1)
    for i, item in catalog.items():
        for genre in item['genres']:
            if genre in GENRES:
                metadata[i, GENRES.index(genre)] = 1
        metadata[i, -1] = (item['year']-1900)/100
    return metadata


def intent_vector(state):
    return [float(g in state['include_genres'])-float(g in state['exclude_genres']) for g in GENRES] + [
        (state['year_min']-1900)/100 if state['year_min'] is not None else 0.,
        (state['year_max']-1900)/100 if state['year_max'] is not None else 1.1]


def sequence_examples(catalog, excluded_users, seed, limit=9000):
    rows = pq.read_table(ROOT/'optimized/data/processed/ml-1m-v1/train.parquet').to_pylist()
    item_ids = {v['raw_id']: i for i, v in catalog.items()}
    users = defaultdict(list)
    for row in rows:
        if row['raw_user_id'] not in excluded_users and row['rating'] >= 4 and row['raw_item_id'] in item_ids:
            users[row['raw_user_id']].append((row['timestamp'], item_ids[row['raw_item_id']]))
    examples = []
    rng = random.Random(seed)
    for user, sequence in sorted(users.items()):
        seq = [i for _, i in sorted(sequence)]
        positions = list(range(5, len(seq)))
        rng.shuffle(positions)
        for pos in positions[:4]:
            history = seq[pos-5:pos]
            target = seq[pos]
            # Up to 3 actual subsequent train positives supervise decoder too.
            continuation = seq[pos:pos+3]
            examples.append((history, target, continuation))
    rng.shuffle(examples)
    return examples[:limit]


def train_generator(directory, catalog, n_items, excluded_users, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CatalogGenerator(n_items, metadata_tensor(n_items, catalog))
    examples = sequence_examples(catalog, excluded_users, seed)
    if not examples:
        raise ValueError('No train-only sequence examples')
    histories = torch.tensor([h for h, _, _ in examples])
    targets = torch.tensor([t for _, t, _ in examples])
    mask = torch.zeros(n_items+1, dtype=torch.bool); mask[list(catalog)] = True
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    curves = []
    neutral = intent_vector(empty_state())
    steps = 0
    for epoch in range(4):
        order = torch.randperm(len(examples))
        total = 0.
        for start in range(0, len(order), 96):
            idx = order[start:start+96]
            hidden = model.encode(histories[idx], torch.tensor([neutral]*len(idx)))
            logits = model.logits(hidden).masked_fill(~mask, -1e9)
            loss = nn.functional.cross_entropy(logits, targets[idx])
            # Train decoder on chronological continuation, not a hand-made slate.
            hidden = model.step(hidden, targets[idx])
            second = torch.tensor([examples[i][2][1] if len(examples[i][2]) > 1 else examples[i][1]
                                   for i in idx.tolist()])
            loss = loss + .5 * nn.functional.cross_entropy(model.logits(hidden).masked_fill(~mask, -1e9), second)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            steps += 1; total += float(loss.detach())
        curves.append({'stage': 'sft', 'epoch': epoch, 'mean_batch_loss': total/math.ceil(len(examples)/96)})
        print(json.dumps({'stage':'generator_sft','seed':seed,'epoch':epoch,'loss':curves[-1]['mean_batch_loss']}),flush=True)
    sft = directory/f'generator_sft_{seed}.pt'; torch.save(model.state_dict(), sft)
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    # Actual slate policy-gradient training. Constraint reward uses train public
    # metadata; relevance uses only real subsequent TRAIN interactions.
    for step in range(60):
        chosen_rows = rng.sample(range(len(examples)), 24)
        h = histories[chosen_rows]
        restrictions = [rng.choice([g for g in catalog[examples[i][1]]['genres'] if g in GENRES])
                        if step%2 and any(g in GENRES for g in catalog[examples[i][1]]['genres']) else None
                        for i in chosen_rows]
        states = [dict(empty_state(), include_genres=[g] if g else []) for g in restrictions]
        hidden = model.encode(h, torch.tensor([intent_vector(s) for s in states]))
        available = mask.expand(len(h), -1).clone()
        available.scatter_(1, h, False)
        valid = available.clone()
        for row, genre in enumerate(restrictions):
            if genre:
                valid[row] &= model.metadata[:,GENRES.index(genre)].bool()
        logps, picked = [], []
        for rank in range(10):
            logits = model.logits(hidden).masked_fill(~available, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            logps.append(distribution.log_prob(action)); picked.append(action)
            available = available.clone(); available.scatter_(1,action[:,None],False)
            hidden = model.step(hidden,action)
        slate = torch.stack(picked,1)
        item_valid = valid.gather(1,slate).float()
        complete = item_valid.prod(1)
        relevant = torch.tensor([[float(int(item) in examples[i][2]) for item in items]
                                 for i,items in zip(chosen_rows,slate.tolist())])
        # A complete slate gets 1; auxiliary rewards cannot exceed .10 in total.
        reward = complete + .05*item_valid.mean(1) + .05*relevant.max(1).values
        advantage = reward-reward.mean()
        loss = -(torch.stack(logps,1).sum(1)*advantage.detach()).mean()
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if step%10==0:
            curves.append({'stage':'reinforce','step':step,'loss':float(loss.detach()),
                           'strict_slate_reward':float(complete.mean()),'mean_reward':float(reward.mean())})
    path = directory/f'generator_rl_{seed}.pt'; torch.save(model.state_dict(), path)
    return {'seed':seed,'examples':len(examples),'sft_steps':steps,'rl_steps':60,
            'architecture':'32-dimensional catalog-ID semantic GRU autoregression; not an LLM',
            'training_source':'train.parquet only; chronological positive continuations; formal users excluded',
            'excluded_user_count':len(excluded_users),'curves':curves,
            'sft_checkpoint_sha256':sha(sft),'rl_checkpoint_sha256':sha(path),
            'parameter_l2_change':math.sqrt(sum(float((v-initial[k]).square().sum()) for k,v in model.state_dict().items()))}


class JointRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def ranking_features(retriever, generator, catalog, state, eligible):
    from ..retrieval.adaptive import _cf_scores
    cf, _ = _cf_scores(retriever,eligible,state['liked_item_ids'],None,False)
    query = defaultdict(float)
    for seed in state['liked_item_ids']:
        for term,w in retriever.content_vectors.get(seed,{}).items():
            query[term] += w/max(1,len(state['liked_item_ids']))
    with torch.no_grad():
        hidden = generator.encode(torch.tensor([state['liked_item_ids'][-5:] or [0]]),
                                  torch.tensor([intent_vector(state)]))
        sequence = generator.logits(hidden).squeeze(0)
        probs = torch.softmax(sequence,0)
    liked_genres = set(g for i in state['liked_item_ids'] for g in catalog.get(i,{}).get('genres',[]))
    return [[float(cf.get(i,0)),
             sum(query.get(t,0)*w for t,w in retriever.content_vectors[i].items()),
             math.log1p(retriever.popularity[i])/10,
             float(sequence[i])/10, float(probs[i])*100,
             len(liked_genres.intersection(catalog[i]['genres']))/max(1,len(liked_genres))]
            for i in eligible]


def train_ranker(directory, retriever, catalog, n_items, excluded_users, seed):
    torch.manual_seed(seed)
    generator = CatalogGenerator(n_items,metadata_tensor(n_items,catalog))
    generator.load_state_dict(torch.load(directory/f'generator_sft_{seed}.pt',map_location='cpu'))
    examples = sequence_examples(catalog,excluded_users,seed,limit=360)
    rng = random.Random(seed)
    all_ids = list(catalog)
    groups = []
    for number,(history,target,_) in enumerate(examples):
        allowed = [i for i in all_ids if i not in history and i!=target]
        candidates = [target] + rng.sample(allowed,63)
        state = dict(empty_state(),liked_item_ids=history,seen_item_ids=history)
        groups.append(ranking_features(retriever,generator,catalog,state,candidates))
        if number%120==0:
            print(json.dumps({'stage':'ranker_training_features','seed':seed,'groups':number}),flush=True)
    x = torch.tensor(groups)
    # Standardization from training only; actual nonconstant learned weights.
    mean = x.flatten(0,1).mean(0); scale = x.flatten(0,1).std(0).clamp_min(1e-5)
    x = (x-mean)/scale
    model = JointRanker(); optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    curves=[]
    for step in range(150):
        logits=model(x)
        loss=nn.functional.cross_entropy(logits,torch.zeros(len(x),dtype=torch.long))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step%30==0:curves.append({'step':step,'loss':float(loss.detach())})
    path=directory/f'ranker_{seed}.pt'
    torch.save({'state_dict':model.state_dict(),'mean':mean,'scale':scale},path)
    return {'seed':seed,'train_groups':len(groups),'negatives_per_group':63,'steps':150,
            'features':['train_itemCF','train_text_tfidf','train_popularity','sequence_logit','sequence_probability','genre_overlap'],
            'algorithm':'learned candidate-group softmax ranking, not LLM2Rec/CCFRec reproduction',
            'curves':curves,'checkpoint_sha256':sha(path)}


def load_models(directory,catalog,n_items,seed=42):
    policy=StatePolicy(); policy.load_state_dict(torch.load(directory/f'policy_rl_{seed}.pt',map_location='cpu')); policy.eval()
    sft_policy=StatePolicy(); sft_policy.load_state_dict(torch.load(directory/f'policy_sft_{seed}.pt',map_location='cpu')); sft_policy.eval()
    generators={}
    for stage in ('sft','rl'):
        m=CatalogGenerator(n_items,metadata_tensor(n_items,catalog))
        m.load_state_dict(torch.load(directory/f'generator_{stage}_{seed}.pt',map_location='cpu')); m.eval()
        generators[stage]=m
    saved=torch.load(directory/f'ranker_{seed}.pt',map_location='cpu')
    ranker=JointRanker(); ranker.load_state_dict(saved['state_dict']); ranker.eval()
    return policy,sft_policy,generators,ranker,saved['mean'],saved['scale']
