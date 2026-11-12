"""Portable entry points for the supplementary reproducibility package."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,hashlib,json,sys,urllib.request,base64,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'/'temporal'))
COMMIT='0fa423c5704de46a8cf0d9b8b542c694c0ae2276'
RAW_SHA='b1a4ce4b90cb76fef5e47b4037f3aa86d8f43e57990c185187df0466f4a14e3d'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonout(path,value):Path(path).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
def download(args):
    dest=Path(args.raw);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if sha(dest)!=RAW_SHA:raise ValueError('Existing raw file differs; use another destination. No file overwritten.')
        print('Existing raw SHA256 verified');return
    def get(url):
        req=urllib.request.Request(url,headers={'User-Agent':'Beach-volleyball-reproducibility','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(req,timeout=90) as r:return r.read()
    relative='data/full_archive/full_archive.csv'
    url=f'https://raw.githubusercontent.com/BigTimeStats/beach-volleyball/{COMMIT}/{relative}'
    try:data=get(url)
    except Exception:
        info=json.loads(get(f'https://api.github.com/repos/BigTimeStats/beach-volleyball/contents/{relative}?ref={COMMIT}'))
        blob=json.loads(get(info['git_url']));data=base64.b64decode(blob['content'])
    if hashlib.sha256(data).hexdigest()!=RAW_SHA:raise ValueError('Downloaded raw SHA256 differs; refusing input')
    dest.write_bytes(data);print('Downloaded pinned raw archive and verified SHA256')
def verify(args):
    import pandas as pd,numpy as np
    import chronology_sensitivity as c
    rawpath=Path(args.raw);assert sha(rawpath)==RAW_SHA,'Raw file hash mismatch'
    raw=c.read_csv(rawpath);d=c.prepare(raw)
    assert raw.shape==(85509,65) and len(d)==85509
    assert int(raw.duplicated().sum())==0
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    counts=raw.groupby(['year','circuit','gender'],dropna=False).agg(n_matches=('date','size'),source_date_min=('date','min'),source_date_max=('date','max')).reset_index()
    counts.to_csv(out/'year_circuit_gender_counts.csv',index=False)
    years=raw.groupby('year').size().rename('n_matches');years.to_csv(out/'year_counts.csv')
    assert int(years.loc[2022])==4206
    split=[int((d.date<=pd.Timestamp('2017-12-31')).sum()),int(((d.date>pd.Timestamp('2017-12-31'))&(d.date<=pd.Timestamp('2019-12-31'))).sum()),int((d.date>pd.Timestamp('2019-12-31')).sum())]
    assert split==[65057,12369,8083]
    c.audit(d,out)
    jsonout(out/'cohort_verification.json',dict(raw_sha256=sha(rawpath),n_rows=len(raw),n_columns=len(raw.columns),full_row_duplicates=int(raw.duplicated().sum()),source_date_min=str(d.date.min().date()),source_date_max=str(d.date.max().date()),splits=split,source_defined_event_gender_units=int(d._event_id.nunique()),historical_download_date='unknown',actual_match_chronology_verified=False,encoding_read_policy='UTF-8 first; latin-1 fallback only on UnicodeDecodeError; no silent row dropping'))
    print('Raw schema, SHA256, cohort, year counts and source-defined events verified')
def build(args):
    import pandas as pd,numpy as np
    import chronology_sensitivity as c
    from portable_frozen import build_frozen_release
    assert sha(args.raw)==RAW_SHA,'Raw file hash mismatch'
    raw=c.read_csv(args.raw);canonical=c.prepare(raw)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    original=c.build_dataset(raw)
    assert len(original)==len(canonical)==85509
    original['_raw_row']=canonical._raw_row.to_numpy();original['_event_id']=canonical._event_id.to_numpy()
    original.to_csv(out/'original_features.csv',index=False)
    audit=dict(n_rows=len(original),model_features=len(c.FEATURES),no_date_model_features=len(c.no_date_features()),raw_sha256=sha(args.raw),exact_csv_serialization_match_required=False)
    if args.reference:
        ref=c.read_csv(args.reference)
        ref.date=pd.to_datetime(ref.date)
        pd.testing.assert_frame_equal(original[ref.columns].reset_index(drop=True),ref.reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
        audit['archived_feature_values_match_with_tolerance']=True
        audit['reference_sha256']=sha(args.reference)
    if not args.original_only:
        for lag in [0,30]:
            f,log=build_frozen_release(canonical,lag)
            assert f._raw_row.tolist()==canonical._raw_row.tolist()
            f.to_csv(out/f'frozen{lag}_features.csv',index=False)
            log.to_csv(out/f'lag{lag}_availability.csv',index=False)
    jsonout(out/'feature_reconstruction_audit.json',audit);print('Feature reconstruction complete')
def train(args):
    import pandas as pd,numpy as np
    from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,log_loss,brier_score_loss
    from catboost import CatBoostClassifier
    import chronology_sensitivity as c
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    frame=c.read_csv(args.features);frame.date=pd.to_datetime(frame.date)
    frame=frame.sort_values(['date','match_num'],kind='mergesort').reset_index(drop=True)
    parts=[frame[frame.date<=pd.Timestamp('2017-12-31')].copy(),frame[(frame.date>pd.Timestamp('2017-12-31'))&(frame.date<=pd.Timestamp('2019-12-31'))].copy(),frame[frame.date>pd.Timestamp('2019-12-31')].copy()]
    if args.smoke:
        parts=[p.head(100).copy() for p in parts];iterations=8;seeds=[42]
    else:
        assert list(map(len,parts))==[65057,12369,8083];iterations=900;seeds=args.seeds
    features=c.no_date_features() if args.no_date_proxies else c.FEATURES
    X=[p[features].copy() for p in parts];cats=[]
    for col in features:
        if pd.api.types.is_numeric_dtype(parts[0][col]):
            fill=parts[0][col].median()
            for x in X:x[col]=x[col].fillna(fill)
        else:
            cats.append(features.index(col));mode=parts[0][col].mode();fill=mode.iloc[0] if len(mode) else 'missing'
            for x in X:x[col]=x[col].fillna(fill).astype(str)
    rows=[];preds=[]
    config=dict(iterations=iterations,depth=6,learning_rate=.03,l2_leaf_reg=5,subsample=.85,loss_function='Logloss',eval_metric='AUC',allow_writing_files=False,thread_count=args.threads,verbose=False)
    jsonout(out/'run_config.json',dict(scenario=args.scenario,seeds=seeds,features=features,parameters=config,has_time='not explicitly passed; library default',smoke_only=args.smoke,feature_file_sha256=sha(args.features)))
    for seed in seeds:
        m=CatBoostClassifier(random_seed=seed,**config)
        m.fit(X[0],parts[0].y.astype(int),cat_features=cats,eval_set=(X[1],parts[1].y.astype(int)),use_best_model=True,verbose=False)
        m.save_model(str(out/f"model_seed{seed}.cbm"))
        jsonout(out/f"model_seed{seed}_all_params.json",m.get_all_params())
        jsonout(out/f"model_seed{seed}_training_metadata.json",json.loads(m.get_metadata()["params"]))
        row=dict(scenario=args.scenario,seed=seed,n_features=len(features),best_iteration=m.get_best_iteration(),smoke_only=args.smoke)
        for split,p,x in [('valid',parts[1],X[1]),('test',parts[2],X[2])]:
            y=p.y.to_numpy(int);prob=m.predict_proba(x)[:,1]
            for k,v in dict(auc=roc_auc_score(y,prob),accuracy=accuracy_score(y,prob>=.5),f1=f1_score(y,prob>=.5),logloss=log_loss(y,prob),brier=brier_score_loss(y,prob)).items():row[f'{split}_{k}']=float(v)
            preds.append(pd.DataFrame(dict(scenario=args.scenario,seed=seed,split=split,raw_row=p._raw_row.to_numpy(),event_id=p._event_id.to_numpy(),y=y,probability=prob)))
        rows.append(row);pd.DataFrame(rows).to_csv(out/'metrics_by_seed.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(out/'predictions.csv',index=False)
        print(f'{args.scenario} seed={seed} test_auc={row["test_auc"]:.8f}',flush=True)
    pd.DataFrame(rows).groupby('scenario').agg({k:['mean','std'] for k in rows[0] if k.startswith(('valid_','test_'))}).to_csv(out/'mean_sd.csv')
def train_main(args):
    # Refuse silent omission of optional model families in the historical script.
    import xgboost,catboost
    subprocess.run([sys.executable,str(ROOT/'code/controlled48/train_baseline_models_with_sd.py'),'--input',str(Path(args.features).resolve()),'--outdir',str(Path(args.out).resolve()),'--threads',str(args.threads),'--seeds',*[str(v) for v in range(42,133,10)]],check=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('download');a.add_argument('--raw',default='data/full_archive.csv');a.set_defaults(func=download)
    a=sub.add_parser('verify');a.add_argument('--raw',required=True);a.add_argument('--out',default='outputs/cohort');a.set_defaults(func=verify)
    a=sub.add_parser('build');a.add_argument('--raw',required=True);a.add_argument('--out',default='outputs/features');a.add_argument('--reference');a.add_argument('--original-only',action='store_true');a.set_defaults(func=build)
    a=sub.add_parser('train-r1');a.add_argument('--features',required=True);a.add_argument('--scenario',required=True);a.add_argument('--out',required=True);a.add_argument('--no-date-proxies',action='store_true');a.add_argument('--smoke',action='store_true');a.add_argument('--seeds',nargs='+',type=int,default=list(range(42,133,10)));a.add_argument('--threads',type=int,default=48);a.set_defaults(func=train)
    a=sub.add_parser('train-main');a.add_argument('--features',required=True);a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=48);a.set_defaults(func=train_main)
    a=sub.add_parser('tests');a.set_defaults(func=lambda args:subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'code'/'temporal'),'-p','test_temporal_invariants.py','-v'],check=True))
    args=p.parse_args();args.func(args)
if __name__=='__main__':main()
