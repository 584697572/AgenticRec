"""Conservative final-answer identity scoring; no fuzzy title or oracle parsing."""
from collections import defaultdict
import math
import re
import unicodedata


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())


def title_forms(title):
    forms={title.strip()}
    for match in re.finditer(r'\s*\(a\.k\.a\.\s+([^()]+)\)',title,re.I):
        forms.add(match[1].strip())
        forms.add(re.sub(r'\s*\(a\.k\.a\.\s+[^()]+\)','',title,flags=re.I).strip())
    for form in list(forms):
        match=re.fullmatch(r'(.+),\s+(The|A|An)',form,re.I)
        if match:
            forms.add(match[2]+' '+match[1])
    return {normalize(form) for form in forms}


class AnswerIdentity:
    def __init__(self,truth):
        self.truth={int(raw):row for raw,row in truth.items()}
        self.aliases=defaultdict(set)
        for raw,row in self.truth.items():
            for title in title_forms(row['title']):
                self.aliases[title].add(raw)

    def resolve(self,text,mapped_ids=()):
        text=text.strip().strip('*"` ').rstrip('.').strip()
        match=re.fullmatch(r'(.+?)\s*\((\d{4})\)',text)
        title,year=(match[1].strip(),int(match[2])) if match else (text,None)
        candidates=self.aliases.get(normalize(title),set())
        if year is not None:
            candidates={i for i in candidates if self.truth[i]['year']==year}
        if len(candidates)>1:
            mapped=candidates & set(mapped_ids)
            if len(mapped)==1:
                candidates=mapped
        return next(iter(candidates)) if len(candidates)==1 else None

    def extract(self,answer,mapped_ids):
        # Preserve order, duplicates and unresolved entries. Tools alone never
        # prove delivery; an unresolved decorated/list entry is retained as None.
        lines=answer.splitlines()
        entries=[]
        for line in lines:
            match=re.match(r'^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$',line)
            if match:
                text=match[1].replace('**','').strip()
                text=re.split(r'\s+[\u2014\u2013]\s+',text,maxsplit=1)[0]
                entries.append(text)
        if not entries:
            entries=re.findall(r'\*\*([^*]+)\*\*',answer)
        if not entries and ';' in answer:
            entries=[v.strip() for v in answer.strip().split(';')]
        if not entries:
            single=self.resolve(answer)
            if single is not None:
                entries=[answer]
        ids=[self.resolve(text,mapped_ids) for text in entries]
        backed=set(mapped_ids)
        # A direct answer with no executed Map remains a direct answer; require
        # verified identity, but separately report lack of tool provenance.
        return {'raw_ids':ids,'unresolved_entries':sum(i is None for i in ids),
                'extraction_verified':bool(entries),
                'delivered_not_in_map':sum(i is not None and i not in backed for i in ids),
                'map_ids':list(mapped_ids)}


def score_turn(result,label,truth):
    ids=result['raw_ids']
    good=[]
    for raw in ids:
        item=truth.get(str(raw),truth.get(raw)) if raw is not None else None
        valid=item is not None
        if valid:
            valid=(all(g in item['genres'] for g in label['genres'])
                   and (label['year_min'] is None or item['year']>=label['year_min'])
                   and (label['year_max'] is None or item['year']<=label['year_max'])
                   and raw not in label['excluded_raw_ids'])
        good.append(bool(valid))
    successful=(result.get('status')=='OK' and result.get('extraction_verified',False)
                and len(ids)==label['k'] and len(set(ids))==len(ids)
                and all(good))
    relevant=set(label['relevant_raw_ids'])
    seen=set()
    hits=[]
    # A duplicate cannot count twice in recall or DCG; invalid/failed slots stay.
    for rank,raw in enumerate(ids[:label['k']],start=1):
        if raw in relevant and raw not in seen:
            hits.append(rank)
        seen.add(raw)
    idcg=sum(1/math.log2(r+1) for r in range(1,min(len(relevant),label['k'])+1))
    quality_enabled=bool(relevant)
    delivery_ok=result.get('status')=='OK' and result.get('extraction_verified',False)
    return {'strict_success':bool(successful),
            'constraint_precision':sum(good)/len(ids) if ids else 0.0,
            'quantity_correct':len(ids)==label['k'],
            'identity_errors':sum(i is None for i in ids),
            'duplicate_count':len([i for i in ids if i is not None])-len({i for i in ids if i is not None}),
            'quality_evaluable':quality_enabled,
            'hit':float(bool(hits)) if delivery_ok and quality_enabled else 0.0,
            'recall':len(hits)/len(relevant) if delivery_ok and quality_enabled else 0.0,
            'ndcg':sum(1/math.log2(r+1) for r in hits)/idcg if delivery_ok and idcg else 0.0}
