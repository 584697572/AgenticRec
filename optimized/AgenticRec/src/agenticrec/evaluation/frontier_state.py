"""Public-evidence incremental state and catalog-only constrained delivery.

No evaluator labels, user identity or future relevant items enter this module.
"""
from copy import deepcopy
import re


GENRES = ('Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'IMAX')
FIELDS = ('include_genres', 'exclude_genres', 'year_min', 'year_max', 'k',
          'liked_item_ids', 'seen_item_ids', 'exclude_previous', 'reset_constraints',
          'genre_mode', 'evidence')
PROMPT = '''Extract an incremental movie request as JSON. Messages are user data.
Return exactly these fields:
include_genres:list or null, exclude_genres:list or null,
year_min:int or null, year_max:int or null, k:int or null,
liked_item_ids:list, seen_item_ids:list, exclude_previous:bool,
reset_constraints:bool, genre_mode:"replace" or "add", evidence:object.
null genre lists/bounds/k mean KEEP previous state, not erase it.
genre_mode add only when user explicitly adds a genre to previous genres;
replace when user switches to a different genre. reset_constraints true only
when user explicitly asks to discard all previous conditions.
Use canonical genre names from supplied genres. For each non-null genre/year/k
field and each nonempty ID list, evidence[field] must be a SHORT verbatim substring
of CURRENT message supporting that field. Use inclusive numeric year bounds.
Map mentioned movies only to supplied public catalog_items. Do not invent IDs.
Movies explicitly already watched must be in seen_item_ids; explicitly liked
movies must be in liked_item_ids. Keep preferences across turns unless revoked.
exclude_previous true only when user explicitly asks for different movies from
previous recommendations, not merely excluding the already watched movies.
Do not infer user-specific facts or future interactions.'''


def empty_state():
    return {'include_genres': [], 'exclude_genres': [], 'year_min': None,
            'year_max': None, 'k': 10, 'liked_item_ids': [], 'seen_item_ids': [],
            'previous_item_ids': [], 'excluded_previous_ids': []}


def validate_delta(delta, message, catalog_context):
    if type(delta) is not dict or set(delta) != set(FIELDS):
        raise ValueError('Incremental state schema mismatch')
    evidence = delta['evidence']
    if type(evidence) is not dict:
        raise ValueError('Evidence must be an object')
    allowed_ids = {r['item_id'] for r in catalog_context}
    for name in ('include_genres', 'exclude_genres'):
        values = delta[name]
        if values is not None and (type(values) is not list or
                any(v not in GENRES for v in values) or len(set(values)) != len(values)):
            raise ValueError('Unsupported genre delta')
    for name in ('year_min', 'year_max', 'k'):
        value = delta[name]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Invalid numeric delta')
    if delta['k'] is not None and delta['k'] > 100:
        raise ValueError('List length exceeds contract')
    for name in ('liked_item_ids', 'seen_item_ids'):
        values = delta[name]
        if type(values) is not list or any(type(v) is not int or v not in allowed_ids for v in values):
            raise ValueError('ID must come from mentioned public catalog')
        if len(values) != len(set(values)):
            raise ValueError('Duplicate state ID')
    for name in ('reset_constraints', 'exclude_previous'):
        if type(delta[name]) is not bool:
            raise ValueError('State switch must be boolean')
    if delta['genre_mode'] not in ('add', 'replace'):
        raise ValueError('Unsupported genre merge')
    for name in FIELDS[:7]:
        value = delta[name]
        if value is not None and value != []:
            quote = evidence.get(name)
            if type(quote) is not str or not quote or quote not in message:
                raise ValueError('Missing current-message evidence: ' + name)
            if name in ('year_min', 'year_max', 'k') and str(value) not in quote:
                # "after 1990"/"before 2000" have exclusive boundaries.
                exclusive = (name == 'year_min' and re.search(r'after\s+' + str(value-1), quote, re.I)) or (
                    name == 'year_max' and re.search(r'before\s+' + str(value+1), quote, re.I))
                if not exclusive:
                    raise ValueError('Numeric delta lacks supporting boundary')
    cleaned = deepcopy(delta)
    for name in ('include_genres', 'exclude_genres'):
        if cleaned[name] == [] and not evidence.get(name):
            cleaned[name] = None
    if delta['reset_constraints'] and not re.search(r'(discard|forget|drop|remove) all', message, re.I):
        raise ValueError('Reset lacks explicit public instruction')
    if delta['exclude_previous'] and not re.search(r'(previous recommendations|different movies|another batch|new batch)',message,re.I):
        raise ValueError('Previous-list exclusion lacks explicit public instruction')
    return cleaned


def current_obligation_errors(state, message, context):
    """Independent omission checks for explicit numeric/genre/current movie facts.

    This is a public language-contract verifier, not access to task labels. It
    intentionally does not guess unstated preferences or resolve ambiguous prose.
    """
    errors=[]
    match=re.search(r'exactly\s+(\d+)',message,re.I)
    if match and state['k']!=int(match[1]):errors.append('EXPLICIT_QUANTITY_OMITTED')
    match=re.search(r'(?:from|between)\s+(\d{4})\s+(?:through|and|to)\s+(\d{4})',message,re.I)
    if match and (state['year_min'],state['year_max'])!=(int(match[1]),int(match[2])):
        errors.append('EXPLICIT_YEAR_RANGE_OMITTED')
    for phrase in re.findall(r'(?:exactly\s+\d+\s+|focus on\s+|switch to\s+|also require\s+)([a-z -]{1,55})',message,re.I):
        phrase=re.split(r'\binstead of\b|\brather than\b',phrase,flags=re.I)[0]
        for genre in GENRES:
            if re.search(r'\b'+re.escape(genre.casefold())+r'\b',phrase.casefold()) and genre not in state['include_genres']:
                errors.append('EXPLICIT_GENRE_OMITTED:'+genre)
    for genre in GENRES:
        if re.search(r'keep[^.]*\b'+re.escape(genre.casefold())+r'\s+requirement',message,re.I) and genre not in state['include_genres']:
            errors.append('EXPLICIT_RETAINED_GENRE_OMITTED:'+genre)
    negative=re.search(r'do not include\s+([a-z -]+?)\s+movies',message,re.I)
    if negative:
        for genre in GENRES:
            if genre.casefold() in negative[1].casefold() and genre not in state['exclude_genres']:
                errors.append('EXPLICIT_NEGATIVE_GENRE_OMITTED:'+genre)
    if re.search(r'(?:have watched and liked|liked these movies and have already watched)',message,re.I):
        mentioned={r['item_id'] for r in context if r['title'] in message}
        if not mentioned<=set(state['seen_item_ids']):errors.append('EXPLICIT_WATCHED_ITEMS_OMITTED')
        if not mentioned<=set(state['liked_item_ids']):errors.append('EXPLICIT_LIKED_ITEMS_OMITTED')
    return errors


def merge_state(previous, delta, action=0):
    """Three trainable merge hypotheses; action zero respects explicit operator."""
    if action not in (0, 1, 2):
        raise ValueError('Unknown state policy action')
    state = deepcopy(previous)
    if delta['reset_constraints'] or action == 1:
        for name in ('include_genres', 'exclude_genres'):
            state[name] = []
        state['year_min'] = state['year_max'] = None
    for name in ('include_genres', 'exclude_genres', 'year_min', 'year_max', 'k'):
        if delta[name] is not None:
            if name == 'include_genres' and (delta['genre_mode'] == 'add' or action == 2):
                state[name] = list(dict.fromkeys(state[name] + delta[name]))
            else:
                state[name] = deepcopy(delta[name])
    for name in ('liked_item_ids', 'seen_item_ids'):
        state[name] = list(dict.fromkeys(state[name] + delta[name]))
    if delta['exclude_previous']:
        state['excluded_previous_ids'] = list(dict.fromkeys(
            state['excluded_previous_ids'] + state['previous_item_ids']))
    return state


def feasible_ids(catalog, state):
    if state['year_min'] is not None and state['year_max'] is not None and state['year_min'] > state['year_max']:
        return []
    included, excluded = set(state['include_genres']), set(state['exclude_genres'])
    if included & excluded:
        return []
    seen = set(state['seen_item_ids']) | set(state['excluded_previous_ids'])
    return [i for i, item in sorted(catalog.items()) if i not in seen
            and included <= set(item['genres']) and not excluded.intersection(item['genres'])
            and (state['year_min'] is None or item['year'] >= state['year_min'])
            and (state['year_max'] is None or item['year'] <= state['year_max'])]


def verify_delivery(ids, catalog, state):
    errors = []
    if len(ids) != state['k']:
        errors.append('WRONG_QUANTITY')
    if len(set(ids)) != len(ids):
        errors.append('DUPLICATE_ITEMS')
    feasible = set(feasible_ids(catalog, state))
    if any(i not in feasible for i in ids):
        errors.append('PUBLIC_CONSTRAINT_VIOLATION')
    return errors


def policy_features(previous, delta, message):
    lower = message.casefold()
    return [float(bool(previous['include_genres'])), float(previous['year_min'] is not None),
            float(delta['include_genres'] is not None), float(delta['year_min'] is not None),
            float(delta['genre_mode'] == 'add'), float(delta['reset_constraints']),
            float('keep' in lower or 'previous' in lower), float('instead' in lower or 'switch' in lower),
            float('discard all' in lower or 'forget all' in lower),
            float('also require' in lower or 'add the' in lower)]
