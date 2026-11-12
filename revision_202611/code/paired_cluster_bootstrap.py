import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap(meta,pred):
    y=meta.y.to_numpy(int)
    defs={'site_year':['circuit','tournament','country','year'],'source_date_event':['circuit','tournament','country','year','date']}
    comparisons=[('catboost','xgboost'),('r1_original','r1_frozen'),('r1_original','r1_without_date_proxies'),('r1_original','r1_buffer30')]
    allrows=[];checks=[]
    for defname,cols in defs.items():
        keys=meta[cols].astype(str).agg('|'.join,axis=1)
        codes,levels=pd.factorize(keys,sort=True);k=len(levels)
        meta.assign(cluster_definition=defname,cluster_key=keys,cluster_code=codes).to_csv(ROOT/f'bootstrap_mapping_{defname}.csv',index=False)
        specs={}
        for name,p in pred.items():
            order=np.argsort(p,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
            specs[name]=(order,starts)
        def aucs(weights):
            wp=weights*y;wn=weights*(1-y);den=wp.sum()*wn.sum()
            if den==0: return None
            out={}
            for name,(order,starts) in specs.items():
                pos=np.add.reduceat(wp[order],starts);neg=np.add.reduceat(wn[order],starts)
                out[name]=float(np.sum(pos*(np.cumsum(neg)-.5*neg))/den)
            return out
        obs=aucs(np.ones(len(y)))
        rng=np.random.default_rng(20261104);rep=[]
        for i in range(5000):
            multiplicity=np.bincount(rng.integers(0,k,size=k),minlength=k)
            weights=multiplicity[codes];v=aucs(weights)
            if v is not None:
                rep.append(v)
                if i<3:
                    for name in pred:
                        ref=roc_auc_score(y,pred[name],sample_weight=weights)
                        assert abs(v[name]-ref)<1e-12
            if (i+1)%1000==0: print(defname,i+1,flush=True)
        draws=pd.DataFrame(rep);draws.to_csv(ROOT/f'bootstrap_replicates_{defname}.csv',index=False)
        for left,right in comparisons:
            delta=draws[left]-draws[right];lo,hi=np.quantile(delta,[.025,.975])
            allrows.append(dict(cluster_definition=defname,n_clusters=k,n_matches=len(y),left=left,right=right,auc_left=obs[left],auc_right=obs[right],delta_left_minus_right=obs[left]-obs[right],ci95_low=lo,ci95_high=hi,valid_replicates=len(draws),requested_replicates=5000,seed=20261104))
        for name in pred:
            lo,hi=np.quantile(draws[name],[.025,.975])
            checks.append(dict(cluster_definition=defname,model=name,auc=obs[name],ci95_low=lo,ci95_high=hi))
    pd.DataFrame(allrows).to_csv(ROOT/'paired_cluster_bootstrap.csv',index=False)
    pd.DataFrame(checks).to_csv(ROOT/'clustered_auc_intervals.csv',index=False)
    return allrows

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Recompute paired cluster bootstrap from bundled fixed predictions (5000 draws per definition).')
    p.add_argument('--input',default=str(Path(__file__).resolve().parents[1]/'reference_results/cluster_bootstrap/test_predictions_and_metadata.csv'))
    p.add_argument('--out',required=True)
    a=p.parse_args();ROOT=Path(a.out);ROOT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(a.input)
    assert len(meta)==8083 and meta._raw_row.nunique()==8083
    names=['catboost','xgboost','r1_original','r1_frozen','r1_without_date_proxies','r1_buffer30']
    bootstrap(meta,{k:meta[k].to_numpy() for k in names})
