"""Training-only time boundaries and corrected supervision, public-only scoring."""
from collections import defaultdict
import torch

DAY = 86400


def temporal_examples(rows,confirmed):
    sequences = defaultdict(list)
    for r in rows:
        if r['rating']<4: raise ValueError('Positive train rows only')
        sequences[r['user_id']].append((r['timestamp'],r['item_id']))
    examples = []; keys = set()
    for user,seq in sorted(sequences.items()):
        seq = sorted(seq)
        if len(seq)!=len({i for ts,i in seq}): raise ValueError('Duplicate positive train item')
        visible = [(ts,i) for ts,i in seq if i in confirmed]
        for q in (.25,.5,.75,.8):
            cut = int(len(visible)*q)
            if cut<10: continue
            anchor = visible[cut-1][0]
            prefix = [(ts,i) for ts,i in seq if ts<=anchor]
            history = [i for ts,i in visible if ts<=anchor]
            for gap in (30,90):
                target = sorted(i for ts,i in visible if ts>=anchor+gap*DAY)
                if not target: continue
                for h in (3,5,10):
                    key = (user,anchor,gap,h)
                    if key in keys: continue
                    keys.add(key)
                    examples.append({'training_user_id':user,'anchor':anchor,'gap_days':gap,
                        'liked_item_ids':history[-h:],'relevant_item_ids':target,
                        'masked_item_ids':sorted(i for ts,i in prefix),'kind':'long'})
    return examples


def ordinal_examples(rows):
    seqs = defaultdict(list)
    for r in rows: seqs[r['user_id']].append((r['timestamp'],r['item_id']))
    result = []
    for user,seq in sorted(seqs.items()):
        seq = sorted(seq); cut = int(len(seq)*.8)
        if cut<10 or cut>=len(seq):continue
        for h in (3,5,10):
            result.append({'training_user_id':user,'liked_item_ids':[i for ts,i in seq[cut-h:cut]],
                'relevant_item_ids':sorted(i for ts,i in seq[cut:]),
                'masked_item_ids':sorted(i for ts,i in seq[:cut]),'kind':'ordinal'})
    return result


def corrected_positive_loss(logits,targets,masked):
    if logits.shape!=targets.shape or masked.shape!=logits.shape or not targets.sum(1).gt(0).all():
        raise ValueError('Compatible nonempty supervision required')
    if (targets.bool()&masked).any(): raise ValueError('Target overlaps past positive mask')
    if not torch.isfinite(logits).all(): raise ValueError('Nonfinite training logits')
    value = logits.masked_fill(masked,-1e9)
    return -(torch.log_softmax(value,1)*targets/targets.sum(1,keepdim=True)).sum(1).mean()


def any_positive_loss(logits,positive,allowed):
    if logits.shape!=positive.shape or allowed.shape!=logits.shape or (positive&~allowed).any() or not positive.any(1).all():
        raise ValueError('Valid nonempty candidate targets required')
    return (torch.logsumexp(logits.masked_fill(~allowed,-1e9),1)-
            torch.logsumexp(logits.masked_fill(~positive,-1e9),1)).mean()


def group_weights(examples,long_mass=.75):
    by = defaultdict(lambda:defaultdict(list))
    for n,r in enumerate(examples): by[r['kind']][r['training_user_id']].append(n)
    result = torch.zeros(len(examples),dtype=torch.float64)
    masses = {'ordinal':1-long_mass,'long':long_mass} if len(by)>1 else {next(iter(by)):1.}
    for kind,users in by.items():
        for user,indices in users.items():
            result[indices] = masses[kind]/len(users)/len(indices)
    return result
