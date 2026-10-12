"""Train-only simple baselines for the frozen full-universe evaluation."""
from collections import Counter
import random


def popularity_scores(candidate_ids, train_positive_item_ids):
    counts = (train_positive_item_ids if isinstance(train_positive_item_ids, Counter)
              else Counter(train_positive_item_ids))
    return {item: int(counts[item]) for item in candidate_ids}


def random_ranking(candidate_ids, seen, k, seed, user_id):
    """Stable per-user seed; sampling without replacement after legal filtering."""
    eligible = [i for i in candidate_ids if i not in seen]
    rng = random.Random((seed << 32) + user_id)
    return rng.sample(eligible, min(k, len(eligible)))
