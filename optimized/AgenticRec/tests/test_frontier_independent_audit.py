import importlib.util
from pathlib import Path
import random

from agenticrec.evaluation.frontier_experiment import score


spec=importlib.util.spec_from_file_location('independent_frontier_auditor',
    Path(__file__).resolve().parents[1]/'scripts/audit_frontier_validation.py')
auditor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def test_independent_predicates_and_metrics_match_on_random_failures():
    rng=random.Random(42)
    truth={str(i):{'genres':['Drama'] if i%2 else ['Drama','Horror'],'year':1980+i} for i in range(1,12)}
    label={'k':5,'genres':['Drama'],'exclude_genres':['Horror'],'year_min':1982,'year_max':1989,
           'excluded_raw_ids':[3],'relevant_raw_ids':[1,5,7,9]}
    for _ in range(100):
        result={'status':rng.choice(['OK','ERROR']),'extraction_verified':rng.choice([True,False]),
                'raw_ids':[rng.choice([None]+list(range(1,12))) for _ in range(rng.randrange(0,8))]}
        expected=auditor.independent_scores(result,label,truth)
        actual=score(result,label,truth)
        assert all(actual[key]==value for key,value in expected.items())


def test_duplicate_target_never_counts_twice_in_independent_audit():
    label={'k':2,'genres':[],'year_min':None,'year_max':None,'excluded_raw_ids':[],'relevant_raw_ids':[1,2]}
    result={'status':'OK','extraction_verified':True,'raw_ids':[1,1]}
    value=auditor.independent_scores(result,label,{'1':{'year':1990,'genres':['Drama']}})
    assert not value['strict_success'] and value['recall']==.5
