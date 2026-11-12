"""Reproduce 48-thread fixed-prediction paired cluster-bootstrap results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
def bootstrap():
    meta=pd.read_csv(ARGS.predictions)
    assert len(meta)==8083 and not meta.raw_row.duplicated().any()
    pred={name:meta[name].to_numpy() for name in ['original','frozen','frozen_without_date_proxies','buffer30']}
    y=meta.y.to_numpy(int);spec={}
    for name,v in pred.items():
        order=np.argsort(v,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(v[order]))+1];spec[name]=(order,starts)
    rows=[];intervals=[]
    for defname,cols,expected in [('site_year',['circuit','tournament','country','year'],84),('source_date_event',['circuit','tournament','country','year','date'],99)]:
        keys=meta[cols].astype(str).agg('|'.join,axis=1);codes,levels=pd.factorize(keys,sort=True);k=len(levels);assert k==expected
        meta[['raw_row','y']].assign(cluster_key=keys,cluster_code=codes).to_csv(ROOT/f'bootstrap_mapping_{defname}.csv',index=False)
        def aucs(w):
            pos=w*y;neg=w*(1-y);den=pos.sum()*neg.sum()
            if not den:return None
            return {name:float(np.sum(np.add.reduceat(pos[o],s)*(np.cumsum(np.add.reduceat(neg[o],s))-.5*np.add.reduceat(neg[o],s)))/den) for name,(o,s) in spec.items()}
        observed=aucs(np.ones(len(y)));rng=np.random.default_rng(20261104);draws=[]
        for j in range(5000):
            w=np.bincount(rng.integers(0,k,size=k),minlength=k)[codes];v=aucs(w)
            if v is not None:
                if j<3:
                    for name in pred:assert abs(v[name]-roc_auc_score(y,pred[name],sample_weight=w))<1e-12
                draws.append(v)
        df=pd.DataFrame(draws);df.to_csv(ROOT/f'bootstrap_replicates_{defname}.csv',index=False)
        for name in pred:
            lo,hi=np.quantile(df[name],[.025,.975]);intervals.append(dict(cluster_definition=defname,scenario=name,auc=observed[name],ci95_low=lo,ci95_high=hi))
            if name=='original':continue
            delta=df.original-df[name];lo,hi=np.quantile(delta,[.025,.975])
            rows.append(dict(cluster_definition=defname,n_clusters=k,n_matches=len(y),left='original',right=name,auc_left=observed['original'],auc_right=observed[name],delta_left_minus_right=observed['original']-observed[name],ci95_low=lo,ci95_high=hi,valid_replicates=len(df),seed=20261104))
        print('BOOTSTRAP',defname,'completed',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'paired_cluster_bootstrap.csv',index=False);pd.DataFrame(intervals).to_csv(ROOT/'clustered_auc_intervals.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[1]
    p.add_argument('--predictions',default=str(base/'reference_results/r1_48threads/test_mean_predictions_and_metadata.csv'))
    p.add_argument('--out',default=str(base/'outputs/bootstrap48'))
    ARGS=p.parse_args();ROOT=Path(ARGS.out);ROOT.mkdir(parents=True,exist_ok=True);bootstrap()
