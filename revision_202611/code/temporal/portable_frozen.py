"""Portable source-date release reconstruction; actual completion is unverified."""
from collections import defaultdict
import pandas as pd
import chronology_sensitivity as c
from frozen_feature_core import emit_row,update_row
def build_frozen_release(d,lag):
    state=(defaultdict(c.AvailableHistory),defaultdict(c.AvailableHistory),defaultdict(c.H2HHistory),defaultdict(lambda:1500.),defaultdict(lambda:1500.))
    event_rows={e:b.to_dict('records') for e,b in d.groupby('_event_id',sort=False)}
    available={e:r[0]['date']+pd.Timedelta(days=lag) for e,r in event_rows.items()}
    pending=sorted(event_rows,key=lambda e:(available[e],event_rows[e][0]['date'],e))
    cursor=0;outputs=[];logs=[]
    for date,block in d.groupby('date',sort=True):
        newly=[]
        while cursor<len(pending) and available[pending[cursor]]<date:
            newly.extend(event_rows[pending[cursor]]);cursor+=1
        for row in sorted(newly,key=lambda r:r['_canonical_row']):update_row(row,*state,k_factor=24,player_k_factor=20)
        for row in block.to_dict('records'):
            feature=emit_row(row,*state,blend_weight=.5);feature['_raw_row']=row['_raw_row'];feature['_event_id']=row['_event_id'];outputs.append(feature)
        logs.append(dict(prediction_start_date=date,eligible_event_count=cursor,lag_days=lag,latest_allowed_source_date_exclusive=date-pd.Timedelta(days=lag),latest_admitted_source_date=max((event_rows[e][0]['date'] for e in pending[:cursor]),default=pd.NaT),completion_dates_verified=False))
    log=pd.DataFrame(logs)
    assert (log.latest_admitted_source_date.isna()|(log.latest_admitted_source_date<log.latest_allowed_source_date_exclusive)).all()
    return pd.DataFrame(outputs),log
