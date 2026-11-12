"""Meaningful temporal invariants; synthetic data only, never manuscript results."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from chronology_sensitivity import (FEATURES, prepare, build_frozen, event_table,
                                    AvailableHistory)
from original_builder import build_dataset
from frozen_feature_core import emit_row, update_row
from collections import defaultdict
from original_builder import RollingHistory, H2HHistory


def raw_row(date, number, event='One', reverse=False):
    row = dict(circuit='AVP', tournament=event, country='USA', year=int(date[:4]),
               date=date, gender='M', match_num=number, bracket='Main', round='Round 1')
    for side, names in [('w', ['Alpha', 'Beta']), ('l', ['Gamma', 'Omega'])]:
        if reverse:
            names = ['Gamma', 'Omega'] if side == 'w' else ['Alpha', 'Beta']
        for i, name in enumerate(names, 1):
            row[f'{side}_player{i}'] = name
            row[f'{side}_p{i}_age'] = 25.
            row[f'{side}_p{i}_hgt'] = 75.
            row[f'{side}_p{i}_country'] = 'USA'
        row[f'{side}_rank'] = '1' if names[0] == 'Alpha' else '2'
    return row


class TemporalTests(unittest.TestCase):
    def test_refactored_core_reproduces_original(self):
        raw = pd.DataFrame([raw_row('2016-01-01', 1), raw_row('2016-01-01', 2, reverse=True),
                            raw_row('2016-02-01', 1, 'Two')])
        expected = build_dataset(raw)
        state = (defaultdict(RollingHistory), defaultdict(RollingHistory),
                 defaultdict(H2HHistory), defaultdict(lambda: 1500.), defaultdict(lambda: 1500.))
        rows = []
        for row in prepare(raw).to_dict('records'):
            rows.append(emit_row(row, *state))
            update_row(row, *state)
        assert_frame_equal(expected, pd.DataFrame(rows))

    def test_same_tournament_and_same_start_results_never_used(self):
        raw = pd.DataFrame([raw_row('2016-01-01', 1), raw_row('2016-01-01', 2),
                            raw_row('2016-01-01', 3, 'Other'), raw_row('2016-02-01', 1, 'Later')])
        a, _ = build_frozen(prepare(raw))
        changed = raw.copy()
        changed.loc[0] = raw_row('2016-01-01', 1, reverse=True)
        b, _ = build_frozen(prepare(changed))
        assert_frame_equal(a.loc[:2, FEATURES], b.loc[:2, FEATURES])
        self.assertTrue((a.loc[:2, 'teamA_prior_matches'] == 0).all())
        self.assertEqual(a.loc[3, 'teamA_prior_matches'], 3)
        self.assertNotEqual(a.loc[3, 'avg_player_elo_diff'], b.loc[3, 'avg_player_elo_diff'])

    def test_future_outcomes_do_not_change_past_features(self):
        raw = pd.DataFrame([raw_row('2016-01-01', 1), raw_row('2016-02-01', 1, 'Two')])
        a, _ = build_frozen(prepare(raw))
        raw.loc[1] = raw_row('2016-02-01', 1, 'Two', reverse=True)
        b, _ = build_frozen(prepare(raw))
        assert_frame_equal(a[FEATURES], b[FEATURES])

    def test_verified_end_dates_exclude_overlapping_event(self):
        d = prepare(pd.DataFrame([raw_row('2016-01-01', 1),
                                 raw_row('2016-01-02', 1, 'Overlap'),
                                 raw_row('2016-01-10', 1, 'Later')]))
        cal = event_table(d)
        cal['end_date'] = ['2016-01-03', '2016-01-04', '2016-01-12']
        cal['evidence_url'] = 'synthetic-test-only'
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'calendar.csv'
            cal.to_csv(p, index=False)
            x, log = build_frozen(d, p)
        self.assertEqual(x.loc[1, 'teamA_prior_matches'], 0)
        self.assertEqual(x.loc[2, 'teamA_prior_matches'], 2)
        eligible = log.dropna(subset=['latest_eligible_availability_date'])
        self.assertTrue((eligible.latest_eligible_availability_date < eligible.prediction_start_date).all())

    def test_out_of_order_releases_keep_date_windows_correct(self):
        h = AvailableHistory()
        h.update(pd.Timestamp('2016-03-01'), 0)
        h.update(pd.Timestamp('2016-01-01'), 1)
        f = h.features(pd.Timestamp('2016-03-10'))
        self.assertEqual(f['last30_match_count'], 1)
        self.assertEqual(f['days_since_last_match'], 9)
        self.assertEqual(f['prior_matches'], 2)

    def test_duplicate_calendar_ids_rejected(self):
        d = prepare(pd.DataFrame([raw_row('2016-01-01', 1)]))
        c = event_table(d)
        c = pd.concat([c, c], ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'calendar.csv'
            c.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                build_frozen(d, p)


if __name__ == '__main__':
    unittest.main()
