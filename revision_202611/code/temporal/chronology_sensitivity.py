#!/usr/bin/env python3
"""R1 major comment 1. Audit, rebuild frozen features, compare fixed CatBoost.

No match number is treated as verified playing chronology. Frozen predictors
exclude the entire current tournament and ALL tournaments sharing its start date.
With --calendar, only tournaments ending strictly before the target start date
are eligible; without it, earlier source start dates are a PROXY, not proof of
completion. See README_CN.md before interpreting the outputs.
"""
import argparse
import bisect
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from original_builder import (RollingHistory, H2HHistory, build_dataset,
                              RAW_EXPECTED_COLUMNS, normalize_name)
from frozen_feature_core import emit_row, update_row

ROOT = Path(__file__).resolve().parent
FEATURES = json.loads((ROOT / 'features.json').read_text())
EVENT_KEYS = ['circuit', 'tournament', 'country', 'year', 'date', 'gender']
SEEDS = [41, 51, 61, 71, 81, 91, 101, 111, 121, 131]


def read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', low_memory=False)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(raw):
    missing = set(RAW_EXPECTED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing raw columns: {sorted(missing)}')
    d = raw.copy()
    d['date'] = pd.to_datetime(d['date'], errors='raise').dt.normalize()
    if d['date'].isna().any():
        raise ValueError('Missing source dates; no silent row dropping allowed.')
    if d[EVENT_KEYS].isna().any().any():
        raise ValueError('Missing event identifiers; resolve before grouping tournaments.')
    players = ['w_player1', 'w_player2', 'l_player1', 'l_player2']
    if d[players].isna().any().any():
        raise ValueError('Missing player identities; resolve before feature reconstruction.')
    d['_raw_row'] = np.arange(len(d))
    # Reproduce the archived generator's sorting; NOT claimed as real chronology.
    d = d.sort_values(['date', 'year', 'match_num'], kind='mergesort').reset_index(drop=True)
    d['_canonical_row'] = np.arange(len(d))
    def event_id(row):
        vals = [str(row[k]).strip() if k != 'date' else row[k].strftime('%Y-%m-%d')
                for k in EVENT_KEYS]
        return hashlib.sha256(json.dumps(vals, ensure_ascii=False).encode()).hexdigest()[:20]
    d['_event_id'] = [event_id(r) for r in d.to_dict('records')]
    return d


class AvailableHistory(RollingHistory):
    """Histories of eligible completed events, with deterministic source-date ties.

    End-date release can insert an older-starting event after a newer one. Keep
    history sorted by source date and archived row order for recent-N statistics.
    This is a reproducible past-data convention, not verified match chronology.
    """
    def __init__(self):
        self.items = []
        self.order = 0

    def update(self, date, win):
        bisect.insort(self.items, (date.value, self.order, int(win)))
        self.order += 1

    def features(self, current_date):
        vals = [v[2] for v in self.items]
        out = {'prior_matches': len(vals),
               'prior_win_pct': float(np.mean(vals)) if vals else np.nan,
               'is_new': int(not vals)}
        for n in (3, 5, 10, 20):
            out[f'recent{n}_win_pct'] = float(np.mean(vals[-n:])) if vals else np.nan
        for n in (30, 90, 180):
            cutoff = (current_date - pd.Timedelta(days=n)).value
            i = bisect.bisect_left(self.items, (cutoff, -1, -1))
            wins = [t[2] for t in self.items[i:] if t[0] < current_date.value]
            out[f'last{n}_match_count'] = len(wins)
            out[f'last{n}_win_pct'] = float(np.mean(wins)) if wins else np.nan
        out['days_since_last_match'] = ((current_date.value - self.items[-1][0]) // 86400000000000
                                        if self.items else np.nan)
        return out


def event_table(d):
    events = d.groupby('_event_id', sort=False).agg(
        **{k: (k, 'first') for k in EVENT_KEYS},
        n_matches=('_event_id', 'size'),
        distinct_match_numbers=('match_num', 'nunique'))
    events['duplicate_match_number_rows'] = (
        d.duplicated(['_event_id', 'match_num'], keep=False).groupby(d['_event_id']).sum())
    events = events.reset_index().rename(columns={'_event_id': 'event_id', 'date': 'start_date'})
    events['end_date'] = ''
    events['evidence_url'] = ''
    return events


def audit(d, out):
    events = event_table(d)
    events.to_csv(out / 'event_calendar_template.csv', index=False)
    events.groupby('circuit').agg(n_events=('event_id', 'size'),
        n_matches=('n_matches', 'sum'),
        events_with_duplicate_numbers=('duplicate_match_number_rows', lambda x: int((x > 0).sum()))
    ).to_csv(out / 'audit_by_circuit.csv')
    d.groupby(['year', 'circuit', 'gender'], dropna=False).size().rename('n_matches').to_csv(
        out / 'retained_matches_by_year.csv')
    duplicate = d[d.duplicated(['_event_id', 'match_num'], keep=False)]
    duplicate[EVENT_KEYS + ['match_num', 'bracket', 'round']].to_csv(
        out / 'duplicate_match_numbers.csv', index=False)
    # Flag players who appear in two event IDs sharing the same start-date proxy.
    # Different men/women events, qualification, or event aliases require review.
    players = d.melt(id_vars=['_event_id', 'date'],
        value_vars=['w_player1', 'w_player2', 'l_player1', 'l_player2'], value_name='player')
    players['player'] = players['player'].map(normalize_name)
    cross = players.drop_duplicates(['date', 'player', '_event_id']).groupby(
        ['date', 'player'])['_event_id'].agg(lambda x: '|'.join(sorted(set(x))))
    cross[cross.str.contains('|', regex=False)].to_csv(out / 'same_date_cross_event_players.csv')
    # Optional per-match scheduled fields are merely reported, never inferred.
    result = dict(n_matches=len(d), n_events=len(events),
        n_source_dates=int(d.date.nunique()), date_min=str(d.date.min().date()),
        date_max=str(d.date.max().date()),
        n_events_with_duplicate_match_numbers=int((events.duplicate_match_number_rows > 0).sum()),
        match_order_verified=False,
        note='Number uniqueness/monotonicity cannot establish actual playing chronology.',
        event_definition=EVENT_KEYS,
        possible_match_time_fields=[c for c in d if any(s in c.lower()
                                   for s in ['localdate', 'localtime', 'start_time', 'match_date'])])
    (out / 'chronology_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return events


def build_frozen(d, calendar=None, k_factor=24, player_k_factor=20, blend_weight=0.5):
    events = event_table(d)
    verified = calendar is not None
    if verified:
        cal = read_csv(calendar)
        required = {'event_id', 'start_date', 'end_date', 'evidence_url'}
        if not required.issubset(cal.columns) or cal.event_id.duplicated().any():
            raise ValueError('Calendar needs unique event_id, start_date, end_date, evidence_url.')
        merged = events[['event_id', 'start_date']].merge(cal, on='event_id', how='left',
                                                       suffixes=('_source', ''))
        for c in ['start_date', 'end_date']:
            merged[c] = pd.to_datetime(merged[c], errors='raise').dt.normalize()
        if merged[['start_date', 'end_date', 'evidence_url']].isna().any().any():
            raise ValueError('Every event requires verified dates and evidence in strict mode.')
        if merged.evidence_url.astype(str).str.strip().eq('').any():
            raise ValueError('Empty evidence reference in verified calendar.')
        if not merged.start_date.eq(merged.start_date_source).all():
            raise ValueError('Calendar/source start dates differ; harmonize raw data explicitly first.')
        if (merged.end_date < merged.start_date).any():
            raise ValueError('Event end before start.')
        available = dict(zip(merged.event_id, merged.end_date))
    else:
        available = dict(zip(events.event_id, events.start_date))
    # All feature queries happen before any same-start-date tournament is released.
    state = (defaultdict(AvailableHistory), defaultdict(AvailableHistory),
             defaultdict(H2HHistory), defaultdict(lambda: 1500.), defaultdict(lambda: 1500.))
    event_rows = {e: block.to_dict('records') for e, block in d.groupby('_event_id', sort=False)}
    pending = sorted(event_rows, key=lambda e: (available[e], event_rows[e][0]['date'], e))
    cursor = 0
    outputs, logs = [], []
    for date, block in d.groupby('date', sort=True):
        newly = []
        while cursor < len(pending) and available[pending[cursor]] < date:
            newly.extend(event_rows[pending[cursor]])
            cursor += 1
        # Results already available are processed deterministically. Their match
        # numbers are NOT used to allow results into their own tournament features.
        for row in sorted(newly, key=lambda r: r['_canonical_row']):
            update_row(row, *state, k_factor=k_factor, player_k_factor=player_k_factor)
        for row in block.to_dict('records'):
            feature = emit_row(row, *state, blend_weight=blend_weight)
            feature['_raw_row'] = row['_raw_row']
            feature['_event_id'] = row['_event_id']
            outputs.append(feature)
        logs.append({'prediction_start_date': date, 'eligible_event_count': cursor,
                     'excluded_current_or_unfinished_events': len(pending) - cursor,
                     'latest_eligible_availability_date': (available[pending[cursor - 1]]
                                                         if cursor else pd.NaT),
                     'availability_basis': 'verified_end_date' if verified else 'source_start_date_proxy'})
    return pd.DataFrame(outputs), pd.DataFrame(logs)


def compare_features(rebuilt, supplied, out):
    old = read_csv(supplied)
    old['date'] = pd.to_datetime(old['date'])
    old = old.sort_values(['date', 'year', 'match_num'], kind='mergesort').reset_index(drop=True)
    keys = ['date', 'year', 'match_num', 'teamA_key', 'teamB_key', 'y']
    if len(old) != len(rebuilt):
        raise ValueError('Original raw/feature row counts differ; resolve cohort first.')
    for c in keys:
        a, b = rebuilt[c].reset_index(drop=True), old[c]
        ok = a.eq(b) | (a.isna() & b.isna())
        if not ok.all():
            raise ValueError(f'Original feature alignment fails at {c}; do not compare unmatched rows.')
    results = []
    for c in FEATURES:
        a, b = rebuilt[c], old[c]
        if pd.api.types.is_numeric_dtype(a):
            equal = np.isclose(a.to_numpy(float), pd.to_numeric(b).to_numpy(float),
                               rtol=1e-10, atol=1e-10, equal_nan=True)
        else:
            equal = (a.fillna('<MISSING>').astype(str) == b.fillna('<MISSING>').astype(str)).to_numpy()
        results.append({'feature': c, 'n_mismatched': int((~equal).sum())})
    result = pd.DataFrame(results)
    result.to_csv(out / 'baseline_feature_reproduction.csv', index=False)
    if result.n_mismatched.sum():
        raise ValueError('Archived generator does not reproduce original features. See reproduction report.')


def no_date_features():
    return [c for c in FEATURES if not any(s in c for s in ('last30', 'last90', 'last180', 'days_since'))]


def train_comparison(original, frozen, out, args, availability_verified):
    import catboost
    import sklearn
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, brier_score_loss
    (out / 'evaluation_environment.json').write_text(json.dumps({
        'catboost': catboost.__version__, 'scikit_learn': sklearn.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__,
        'python': platform.python_version()}, indent=2))
    scenarios = [('original_sequential', original, FEATURES),
                 ('tournament_frozen', frozen, FEATURES)]
    if args.exclude_date_features:
        scenarios.append(('frozen_without_date_proxies', frozen, no_date_features()))
    rows, predictions = [], []
    for scenario, frame, features in scenarios:
        tr = frame[frame.date <= pd.Timestamp(args.train_end)].copy()
        va = frame[(frame.date > pd.Timestamp(args.train_end)) &
                   (frame.date <= pd.Timestamp(args.valid_end))].copy()
        te = frame[frame.date > pd.Timestamp(args.valid_end)].copy()
        if any(len(x) == 0 or x.y.nunique() < 2 for x in (tr, va, te)):
            raise ValueError('Every split must be nonempty and contain both target classes.')
        # The same event is represented by a single start date and cannot cross splits.
        assert not set(tr._event_id) & (set(va._event_id) | set(te._event_id))
        assert not set(va._event_id) & set(te._event_id)
        X = [x[features].copy() for x in (tr, va, te)]
        cats = []
        for c in features:
            if pd.api.types.is_numeric_dtype(tr[c]):
                median = tr[c].median()
                # Preserve the archived preprocessing, including all-missing NaN.
                for x in X:
                    x[c] = x[c].fillna(median)
            else:
                cats.append(features.index(c))
                mode = tr[c].mode()
                fill = mode.iloc[0] if len(mode) else 'missing'
                for x in X:
                    x[c] = x[c].fillna(fill).astype(str)
        for seed in args.seeds:
            model = CatBoostClassifier(iterations=args.iterations, depth=args.depth,
                learning_rate=args.learning_rate, l2_leaf_reg=args.l2_leaf_reg,
                subsample=args.subsample, loss_function='Logloss', eval_metric='AUC',
                random_seed=seed, allow_writing_files=False, thread_count=args.threads,
                verbose=False)
            # Match the archived no-identity baseline; no new has_time setting,
            # tuning, test-based seed selection, or other model change is introduced.
            model.fit(X[0], tr.y.astype(int), cat_features=cats,
                      eval_set=(X[1], va.y.astype(int)), use_best_model=True, verbose=False)
            result = {'scenario': scenario, 'seed': seed, 'n_features': len(features),
                      'n_train': len(tr), 'n_valid': len(va), 'n_test': len(te),
                      'best_iteration': model.get_best_iteration(),
                      'completion_dates_verified': availability_verified}
            for split, data, xx in [('valid', va, X[1]), ('test', te, X[2])]:
                p = model.predict_proba(xx)[:, 1]
                y = data.y.to_numpy(int)
                for metric, value in dict(auc=roc_auc_score(y, p),
                    accuracy=accuracy_score(y, p >= .5), f1=f1_score(y, p >= .5, zero_division=0),
                    logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p)).items():
                    result[f'{split}_{metric}'] = float(value)
                predictions.append(pd.DataFrame({'scenario': scenario, 'seed': seed,
                    'split': split, 'raw_row': data._raw_row.to_numpy(),
                    'event_id': data._event_id.to_numpy(), 'y': y, 'probability': p}))
            rows.append(result)
            pd.DataFrame(rows).to_csv(out / 'metrics_by_seed.csv', index=False)
            print(f'{scenario} seed={seed}: test AUC={result["test_auc"]:.6f}', flush=True)
    runs = pd.DataFrame(rows)
    metrics = [c for c in runs if c.startswith(('valid_', 'test_'))]
    summary = runs.groupby('scenario', sort=False)[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out / 'summary_mean_sd.csv')
    base = runs[runs.scenario == 'original_sequential'].set_index('seed')
    deltas = []
    for scenario in runs.scenario.unique()[1:]:
        new = runs[runs.scenario == scenario].set_index('seed')
        for seed in args.seeds:
            deltas.append({'scenario': scenario, 'seed': seed,
                           **{f'delta_{c}': new.loc[seed, c] - base.loc[seed, c] for c in metrics}})
    pd.DataFrame(deltas).to_csv(out / 'paired_seed_differences.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / 'predictions.csv', index=False)
    (out / 'evaluation_note.txt').write_text(
        'Mean ± SD describes training-seed variation, not a sampling confidence interval.\n'
        'No test-set-based model/seed/parameter selection is performed.\n'
        'Without a fully sourced end-date calendar, frozen results use earlier-start-date proxies; '
        'they do not certify that all earlier tournaments had completed.\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--original-features')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--stage', choices=['audit', 'build', 'evaluate', 'all'], default='all')
    ap.add_argument('--calendar', help='Completed event_calendar_template.csv with sourced end dates')
    ap.add_argument('--exclude-date-features', action='store_true')
    ap.add_argument('--train-end', default='2017-12-31')
    ap.add_argument('--valid-end', default='2019-12-31')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--iterations', type=int, default=900)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--learning-rate', type=float, default=.03)
    ap.add_argument('--l2-leaf-reg', type=float, default=5.)
    ap.add_argument('--subsample', type=float, default=.85)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--k-factor', type=int, default=24)
    ap.add_argument('--player-k-factor', type=int, default=20)
    ap.add_argument('--blend-weight', type=float, default=.5)
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        ap.error('Seeds must be unique.')
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(vars(args), raw_sha256=digest(args.raw), python=platform.python_version(),
                    pandas=pd.__version__, numpy=np.__version__,
                    completion_dates_verified=bool(args.calendar),
                    baseline_generator_sha256=digest(ROOT / 'original_builder.py'))
    if args.original_features:
        manifest['original_features_sha256'] = digest(args.original_features)
    if args.calendar:
        manifest['calendar_sha256'] = digest(args.calendar)
    path = out / 'run_manifest.json'
    if args.stage == 'evaluate':
        if not path.exists():
            raise ValueError('Run build first; evaluate requires its manifest.')
        previous = json.loads(path.read_text())
        for key in ['raw_sha256', 'calendar_sha256', 'k_factor', 'player_k_factor', 'blend_weight']:
            if previous.get(key) != manifest.get(key):
                raise ValueError(f'Build/evaluate settings differ: {key}')
        original = read_csv(out / 'features_original_rebuilt.csv')
        frozen = read_csv(out / 'features_tournament_frozen.csv')
        for frame in (original, frozen):
            frame['date'] = pd.to_datetime(frame.date)
    else:
        d = prepare(read_csv(args.raw))
        audit(d, out)
        if args.stage == 'audit':
            (out / 'audit_manifest.json').write_text(json.dumps(manifest, indent=2))
            return
        print('Rebuilding archived sequential features...', flush=True)
        original = build_dataset(d, k_factor=args.k_factor,
            player_k_factor=args.player_k_factor, blend_weight=args.blend_weight)
        original['_raw_row'] = d._raw_row.to_numpy()
        original['_event_id'] = d._event_id.to_numpy()
        if args.original_features:
            compare_features(original, args.original_features, out)
        print('Rebuilding tournament-frozen features...', flush=True)
        frozen, log = build_frozen(d, args.calendar, args.k_factor,
                                  args.player_k_factor, args.blend_weight)
        original.to_csv(out / 'features_original_rebuilt.csv', index=False)
        frozen.to_csv(out / 'features_tournament_frozen.csv', index=False)
        log.to_csv(out / 'feature_availability_audit.csv', index=False)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.stage in ('evaluate', 'all'):
        train_comparison(original, frozen, out, args, bool(args.calendar))
    print(f'Completed {args.stage}: {out}', flush=True)


if __name__ == '__main__':
    main()
