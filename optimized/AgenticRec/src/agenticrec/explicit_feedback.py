"""Separate low-rating supervision; unknown sampling retains its old contract."""
import math
import torch
import torch.nn.functional as F


def mapped_lows(rows, user_map, item_map, seen, edges):
    lows = {u: [] for u in seen}
    positive = set(edges)
    unique = set()
    for row in rows:
        user = user_map.get(str(row['raw_user_id']))
        item = item_map.get(str(row['raw_item_id']))
        if user is None or item is None or row['rating'] > 2:
            continue
        if row['rating'] < 1 or user not in lows or item not in seen[user] or (user,item) in positive or (user,item) in unique:
            raise ValueError('Explicit low rating inconsistent with frozen train contract')
        unique.add((user,item)); lows[user].append(item)
    return {u: sorted(items) for u,items in lows.items()}


class LowSampler:
    """Uniform per-user low items, independent RNG; zero means absent feedback."""
    def __init__(self, n_users, n_items, lows, seed):
        if min(n_users,n_items) < 1 or any(type(u) is not int or not 1 <= u <= n_users for u in lows):
            raise ValueError('Invalid low-feedback user IDs')
        for items in lows.values():
            if len(items) != len(set(items)) or any(type(i) is not int or not 1 <= i <= n_items for i in items):
                raise ValueError('Invalid low-feedback item IDs')
        self.counts = torch.tensor([len(lows.get(u,[])) for u in range(1,n_users+1)])
        self.table = torch.zeros(n_users, max(1,int(self.counts.max())), dtype=torch.long)
        for u,items in lows.items():
            self.table[u-1,:len(items)] = torch.tensor(items,dtype=torch.long)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, users):
        if users.dtype != torch.long or users.ndim != 1 or bool(((users < 1)|(users > len(self.counts))).any()):
            raise ValueError('Invalid low-feedback batch users')
        counts = self.counts[users-1]
        index = (torch.rand(len(users),generator=self.generator)*counts).long()
        return self.table[users-1,index]


def explicit_loss(model, kind, users, positive, unknown, low, weight, regularization):
    """Mean unknown BPR + unchanged L2 + weighted batch-masked explicit BPR.

    Absent lows contribute zero (not a last-item or padding example). Both arms
    draw/compute the same extra branch, weight=0 is the exact ablation.
    """
    if kind not in ('bpr','lightgcn') or not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid explicit-feedback objective')
    if kind == 'lightgcn':
        ue,ie = model.propagate()
    else:
        ue,ie = model.users.weight[1:],model.items.weight[1:]
    u = ue[users-1]; p = ie[positive-1]; n = ie[unknown-1]
    margin = (u*p).sum(1)-(u*n).sum(1)
    base = F.softplus(-margin)
    ego_u,ego_p,ego_n = model.users(users),model.items(positive),model.items(unknown)
    reg = ((ego_u.square()+ego_p.square()+ego_n.square()).sum(1)/2).mean()
    mask = low > 0
    # Index only real examples; never index item -1 for users with no low rating.
    if bool(mask.any()):
        low_ids = low[mask]
        low_margin = (u[mask]*p[mask]).sum(1)-(u[mask]*ie[low_ids-1]).sum(1)
        low_data = F.softplus(-low_margin).sum()/len(users)
    else:
        low_data = base.sum()*0
    loss = base.mean()+regularization*reg+weight*low_data
    grad = torch.sigmoid(-margin.detach())
    diagnostic = {'data_loss':float(base.detach().mean()),'margin':float(margin.detach().mean()),
                  'gradient':float(grad.mean()),'easy_fraction':float((grad<.05).float().mean()),
                  'low_loss':float(low_data.detach()),'low_pairs':int(mask.sum())}
    return loss,diagnostic
