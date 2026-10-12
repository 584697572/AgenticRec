import importlib.util
from pathlib import Path


def test_each_catalog_can_confirm_identity_without_intersection_or_genre_equality():
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('correction',root/'optimized/AgenticRec/scripts/correct_comparison_identity.py')
    module=importlib.util.module_from_spec(spec)
    scripts=str(root/'optimized/AgenticRec/scripts')
    import sys
    sys.path.insert(0,scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)

    class Index:
        def resolve(self,title,year):
            return {'status':'PASS','raw_ids':[1196]} if title=='Empire' and year==1980 else {'status':'TITLE_ABSENT','raw_ids':[]}

    original=[{'id':10,'title':'Empire','release_date':1980,'tags':['Drama']}]
    optimized=[{'item_id':20,'raw_item_id':1196,'title':'Empire (1980)','genres':['Action']},
               {'item_id':21,'raw_item_id':1196,'title':'Not Empire (1980)','genres':[]}]
    split=lambda text:(text.rsplit(' (',1)[0],int(text[-5:-1]))
    old,new,_=module.identity_maps(original,optimized,Index(),split)
    assert old=={10:1196}
    assert new=={20:1196}
    # A film absent from the original catalog still has its own source-backed ID.
    old,new,_=module.identity_maps([],optimized,Index(),split)
    assert old=={} and new=={20:1196}


def test_duplicate_original_rows_are_not_guessed():
    root=Path(__file__).resolve().parents[3]
    scripts=root/'optimized/AgenticRec/scripts'
    import sys
    sys.path.insert(0,str(scripts))
    try:
        import correct_comparison_identity as module
    finally:
        sys.path.remove(str(scripts))

    class Index:
        def resolve(self,title,year):
            return {'status':'PASS','raw_ids':[1]}

    old,new,_=module.identity_maps([{'id':1,'title':'Movie','release_date':2000},
                                   {'id':2,'title':'Movie','release_date':2000}],[],Index(),None)
    assert old=={} and new=={}


def test_exact_source_match_precedes_a_broader_ambiguous_alias():
    import sys
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/'optimized/AgenticRec/scripts'))
    try:
        import correct_comparison_identity as module
    finally:
        sys.path.pop(0)

    class Index:
        def resolve(self,title,year):
            return {'status':'PASS','raw_ids':[1 if title=='Exact' else 2]}

    old,new,_=module.identity_maps([{'id':10,'title':'Exact','release_date':2000}],[],
        Index(),None,lambda title:{'Exact','Broader alias'})
    assert old=={10:1}


def test_explicit_source_alias_can_confirm_optimized_movie_without_original_row():
    import sys
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/'optimized/AgenticRec/scripts'))
    try:
        import correct_comparison_identity as module
    finally:
        sys.path.pop(0)

    class Index:
        def resolve(self,title,year):
            return ({'status':'PASS','raw_ids':[2]} if title=='Source alias'
                    else {'status':'TITLE_ABSENT','raw_ids':[]})

    old,new,_=module.identity_maps([], [{'item_id':20,'raw_item_id':2,'title':'Full aka title (2000)'}],
        Index(),lambda title:('Full aka title',2000),lambda title:{title,'Source alias'})
    assert old=={} and new=={20:2}
