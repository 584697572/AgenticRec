from types import SimpleNamespace

import pytest
import torch

from agenticrec.evaluation.sequence_agent import PublicRanker, SequenceAgentSystem
from agenticrec.evaluation.frontier_experiment import FrontierSystem
from agenticrec.evaluation.frontier_state import empty_state
from agenticrec.evaluation.rank_upgrade_models import CFIndex


class FixedSequence(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.observed = []

    def forward(self, history):
        self.observed.append(history.clone())
        return torch.arange(self.n, dtype=torch.float32).unsqueeze(0).expand(len(history), -1)


def environment():
    catalog = {i: {'item_id': i, 'raw_id': 100+i, 'title': 'Movie '+str(i),
                   'year': 1990 if i <= 20 else 2020, 'genres': ['Drama'] if i <= 20 else ['Horror']}
               for i in range(1, 31)}
    # Equal popularity makes independent sequence ordering unambiguous.
    return {'catalog': catalog, 'index': CFIndex(30, [(1, i) for i in catalog])}


def delta(**changes):
    value = {'include_genres': None, 'exclude_genres': None, 'year_min': None, 'year_max': None,
             'k': None, 'liked_item_ids': [], 'seen_item_ids': [], 'exclude_previous': False,
             'reset_constraints': False, 'genre_mode': 'replace', 'evidence': {}}
    value.update(changes)
    return value


def test_constraints_filter_full_sequence_scores_before_selecting_k():
    env = environment(); model = FixedSequence(30)
    state = empty_state()
    state.update(liked_item_ids=[1, 2, 3], seen_item_ids=[1, 2, 3], include_genres=['Drama'],
                 exclude_genres=['Horror'], year_min=1985, year_max=2000, k=5)
    assert PublicRanker('sequence', env, model).rank(state) == [20, 19, 18, 17, 16]
    # Top ten globally all violate the request: post-filtering top ten would fail.
    assert all(i <= 20 for i in PublicRanker('sequence', env, model).rank(state))


def test_sequence_only_uses_last_ten_public_likes_and_ignores_hidden_fields():
    env = environment(); model = FixedSequence(30); ranker = PublicRanker('sequence', env, model)
    state = empty_state(); state.update(liked_item_ids=list(range(1, 13)), seen_item_ids=list(range(1, 13)))
    a = ranker.rank(dict(state, user_id=999, relevant_item_ids=[1]))
    b = ranker.rank(dict(state, user_id=1, relevant_item_ids=[30]))
    assert a == b and model.observed[-1].tolist() == [list(range(3, 13))]


@pytest.mark.parametrize('system', ['sequence', 'itemcf'])
def test_cold_start_uses_same_constrained_popular_fallback(system):
    env = environment(); state = empty_state(); state.update(k=5, include_genres=['Drama'], seen_item_ids=[1])
    assert PublicRanker(system, env, FixedSequence(30)).rank(state) == PublicRanker('popular', env).rank(state)


def test_live_hook_reuses_actual_parser_and_keeps_executed_multiturn_state():
    assert SequenceAgentSystem.parse is FrontierSystem.parse and SequenceAgentSystem.turn is FrontierSystem.turn
    env = environment(); system = SequenceAgentSystem(None, PublicRanker('sequence', env, FixedSequence(30)))
    public = {'messages': ['first request', 'another batch'], 'catalog_context': []}
    first = system.execute(public, 0, delta(liked_item_ids=[1, 2, 3], seen_item_ids=[1, 2, 3],
                          include_genres=['Drama'], year_min=1985, year_max=2000, k=5))
    second = system.execute(public, 1, delta(exclude_previous=True))
    assert first['status'] == second['status'] == 'OK'
    assert not set(first['raw_ids'])&set(second['raw_ids'])
    assert second['state']['year_min'] == 1985 and second['state']['liked_item_ids'] == [1, 2, 3]
    assert system.history[-1]['content'] == second['answer']
    assert second['raw_ids'] == [env['catalog'][i]['raw_id'] for i in second['mapped_optimized_ids']]


def test_impossible_conditions_do_not_deliver_out_of_bounds_movies():
    env = environment(); system = SequenceAgentSystem(None, PublicRanker('sequence', env, FixedSequence(30)))
    result = system.execute({'messages': ['unanswerable']}, 0, delta(year_min=2030, year_max=2031))
    assert result['status'] == 'CONSTRAINT_FAILURE' and result['eligible_count'] == 0 and result['raw_ids'] == []


def test_nonfinite_checkpoint_logits_are_rejected_before_delivery():
    class Broken(FixedSequence):
        def forward(self, history): return torch.full((1, self.n), float('nan'))
    state = empty_state(); state.update(liked_item_ids=[1])
    with pytest.raises(ValueError): PublicRanker('sequence', environment(), Broken(30)).rank(state)
