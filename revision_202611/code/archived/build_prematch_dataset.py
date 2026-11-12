
import argparse
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


RAW_EXPECTED_COLUMNS = [
    'circuit', 'tournament', 'country', 'year', 'date', 'gender', 'match_num',
    'w_player1', 'w_p1_age', 'w_p1_hgt', 'w_p1_country', 'w_player2', 'w_p2_age', 'w_p2_hgt', 'w_p2_country', 'w_rank',
    'l_player1', 'l_p1_age', 'l_p1_hgt', 'l_p1_country', 'l_player2', 'l_p2_age', 'l_p2_hgt', 'l_p2_country', 'l_rank',
    'bracket', 'round'
]


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return 'unknown'
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)
    return name.lower()


def pretty_name(name: str) -> str:
    if pd.isna(name):
        return 'Unknown'
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)
    return name


def safe_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_rank(rank_value):
    if pd.isna(rank_value):
        return np.nan, 0, np.nan
    text = str(rank_value).strip()
    if not text:
        return np.nan, 0, np.nan

    main_match = re.search(r'(\d+)', text)
    rank_main = float(main_match.group(1)) if main_match else np.nan

    q_match = re.search(r'Q\s*(\d+)', text, flags=re.IGNORECASE)
    is_qualifier = 1 if q_match else 0
    qualifier_num = float(q_match.group(1)) if q_match else np.nan
    return rank_main, is_qualifier, qualifier_num


class RollingHistory:
    def __init__(self):
        self.matches = 0
        self.wins = 0
        self.recent3 = deque(maxlen=3)
        self.recent5 = deque(maxlen=5)
        self.recent10 = deque(maxlen=10)
        self.recent20 = deque(maxlen=20)
        self.time_queue = deque()  # (date, win)
        self.last_date = None

    def _prune(self, current_date):
        if pd.isna(current_date):
            return
        while self.time_queue:
            d = self.time_queue[0][0]
            if pd.isna(d):
                self.time_queue.popleft()
                continue
            if (current_date - d).days > 180:
                self.time_queue.popleft()
            else:
                break

    def _window_stats(self, current_date, days):
        if pd.isna(current_date):
            return np.nan, np.nan
        vals = [w for d, w in self.time_queue if pd.notna(d) and (current_date - d).days <= days]
        if len(vals) == 0:
            return 0, np.nan
        return len(vals), float(np.mean(vals))

    def features(self, current_date):
        self._prune(current_date)
        prior_win_pct = self.wins / self.matches if self.matches > 0 else np.nan

        count30, win30 = self._window_stats(current_date, 30)
        count90, win90 = self._window_stats(current_date, 90)
        count180, win180 = self._window_stats(current_date, 180)

        if self.last_date is None or pd.isna(current_date) or pd.isna(self.last_date):
            days_since = np.nan
        else:
            days_since = (current_date - self.last_date).days

        return {
            'prior_matches': self.matches,
            'prior_win_pct': prior_win_pct,
            'recent3_win_pct': float(np.mean(self.recent3)) if len(self.recent3) > 0 else np.nan,
            'recent5_win_pct': float(np.mean(self.recent5)) if len(self.recent5) > 0 else np.nan,
            'recent10_win_pct': float(np.mean(self.recent10)) if len(self.recent10) > 0 else np.nan,
            'recent20_win_pct': float(np.mean(self.recent20)) if len(self.recent20) > 0 else np.nan,
            'last30_match_count': count30,
            'last30_win_pct': win30,
            'last90_match_count': count90,
            'last90_win_pct': win90,
            'last180_match_count': count180,
            'last180_win_pct': win180,
            'days_since_last_match': days_since,
            'is_new': int(self.matches == 0),
        }

    def update(self, date, win: int):
        self.matches += 1
        self.wins += int(win)
        self.recent3.append(int(win))
        self.recent5.append(int(win))
        self.recent10.append(int(win))
        self.recent20.append(int(win))
        self.time_queue.append((date, int(win)))
        self.last_date = date


class H2HHistory:
    def __init__(self):
        self.matches = 0
        self.a_wins = 0
        self.recent3 = deque(maxlen=3)

    def features(self):
        a_win_pct = self.a_wins / self.matches if self.matches > 0 else np.nan
        recent3_a_win_pct = float(np.mean(self.recent3)) if len(self.recent3) > 0 else np.nan
        return {
            'h2h_prior_matches': self.matches,
            'h2h_a_win_pct': a_win_pct,
            'h2h_recent3_a_win_pct': recent3_a_win_pct,
        }

    def update(self, a_win: int):
        self.matches += 1
        self.a_wins += int(a_win)
        self.recent3.append(int(a_win))


def elo_expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def elo_update(ra, rb, a_win, k=24):
    ea = elo_expected(ra, rb)
    sa = float(a_win)
    new_ra = ra + k * (sa - ea)
    new_rb = rb + k * ((1.0 - sa) - (1.0 - ea))
    return new_ra, new_rb


def get_team_dict(row, side_prefix: str):
    p1 = pretty_name(row[f'{side_prefix}_player1'])
    p2 = pretty_name(row[f'{side_prefix}_player2'])
    p1_norm = normalize_name(p1)
    p2_norm = normalize_name(p2)
    ordered_players = sorted([p1_norm, p2_norm])
    ordered_players_pretty = sorted([p1, p2], key=lambda x: normalize_name(x))
    team_key = '__'.join(ordered_players)

    rank_main, is_qualifier, qualifier_num = parse_rank(row[f'{side_prefix}_rank'])
    age1 = safe_float(row[f'{side_prefix}_p1_age'])
    age2 = safe_float(row[f'{side_prefix}_p2_age'])
    hgt1 = safe_float(row[f'{side_prefix}_p1_hgt'])
    hgt2 = safe_float(row[f'{side_prefix}_p2_hgt'])
    avg_age = np.nanmean([age1, age2])
    avg_hgt = np.nanmean([hgt1, hgt2])
    c1 = row.get(f'{side_prefix}_p1_country', np.nan)
    c2 = row.get(f'{side_prefix}_p2_country', np.nan)
    same_country = int(pd.notna(c1) and pd.notna(c2) and str(c1).strip() == str(c2).strip())

    return {
        'team_key': team_key,
        'player1': ordered_players_pretty[0],
        'player2': ordered_players_pretty[1],
        'player1_key': ordered_players[0],
        'player2_key': ordered_players[1],
        'avg_age': avg_age,
        'avg_hgt': avg_hgt,
        'rank_main': rank_main,
        'is_qualifier': is_qualifier,
        'qualifier_num': qualifier_num,
        'same_country': same_country,
    }


def summarize_player_history(team, player_hist, current_date):
    p1 = player_hist[team['player1_key']].features(current_date)
    p2 = player_hist[team['player2_key']].features(current_date)

    def avg(key):
        vals = [p1[key], p2[key]]
        vals = [v for v in vals if pd.notna(v)]
        if not vals:
            return np.nan
        return float(np.mean(vals))

    return {
        'avg_player_prior_matches': avg('prior_matches'),
        'avg_player_prior_win_pct': avg('prior_win_pct'),
        'avg_player_recent10_win_pct': avg('recent10_win_pct'),
        'avg_player_recent20_win_pct': avg('recent20_win_pct'),
        'avg_player_last90_match_count': avg('last90_match_count'),
        'avg_player_last90_win_pct': avg('last90_win_pct'),
        'avg_player_days_since_last_match': avg('days_since_last_match'),
        'both_players_new': int(p1['prior_matches'] == 0 and p2['prior_matches'] == 0),
    }


def diff_or_nan(a, b):
    if pd.notna(a) and pd.notna(b):
        return a - b
    return np.nan


def build_dataset(df: pd.DataFrame, k_factor: int = 24, player_k_factor: int = 20, blend_weight: float = 0.5) -> pd.DataFrame:
    for col in RAW_EXPECTED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['date', 'year', 'match_num'], kind='mergesort').reset_index(drop=True)

    team_hist = defaultdict(RollingHistory)      # pair history
    player_hist = defaultdict(RollingHistory)    # individual history
    h2h_hist = defaultdict(H2HHistory)
    team_elo = defaultdict(lambda: 1500.0)
    player_elo = defaultdict(lambda: 1500.0)

    rows = []

    for _, row in df.iterrows():
        current_date = row['date']
        w = get_team_dict(row, 'w')
        l = get_team_dict(row, 'l')

        if w['team_key'] <= l['team_key']:
            teamA, teamB = w, l
            y = 1
        else:
            teamA, teamB = l, w
            y = 0

        histA = team_hist[teamA['team_key']].features(current_date)
        histB = team_hist[teamB['team_key']].features(current_date)

        playerA = summarize_player_history(teamA, player_hist, current_date)
        playerB = summarize_player_history(teamB, player_hist, current_date)

        pair_key = '||'.join(sorted([teamA['team_key'], teamB['team_key']]))
        pair_hist = h2h_hist[pair_key].features()
        if teamA['team_key'] <= teamB['team_key']:
            h2h_a_win_pct = pair_hist['h2h_a_win_pct']
            h2h_recent3_a_win_pct = pair_hist['h2h_recent3_a_win_pct']
        else:
            h2h_a_win_pct = np.nan if pd.isna(pair_hist['h2h_a_win_pct']) else 1.0 - pair_hist['h2h_a_win_pct']
            h2h_recent3_a_win_pct = np.nan if pd.isna(pair_hist['h2h_recent3_a_win_pct']) else 1.0 - pair_hist['h2h_recent3_a_win_pct']

        eloA = team_elo[teamA['team_key']]
        eloB = team_elo[teamB['team_key']]

        a_p1_elo = player_elo[teamA['player1_key']]
        a_p2_elo = player_elo[teamA['player2_key']]
        b_p1_elo = player_elo[teamB['player1_key']]
        b_p2_elo = player_elo[teamB['player2_key']]
        a_avg_player_elo = (a_p1_elo + a_p2_elo) / 2.0
        b_avg_player_elo = (b_p1_elo + b_p2_elo) / 2.0
        a_blend_elo = blend_weight * eloA + (1.0 - blend_weight) * a_avg_player_elo
        b_blend_elo = blend_weight * eloB + (1.0 - blend_weight) * b_avg_player_elo

        out = {
            'date': row['date'],
            'year': row['year'],
            'month': row['date'].month if pd.notna(row['date']) else np.nan,
            'gender': row['gender'],
            'circuit': row['circuit'],
            'tournament': row['tournament'],
            'country': row['country'],
            'bracket': row['bracket'],
            'round': row['round'],
            'match_num': row['match_num'],
            'teamA_key': teamA['team_key'],
            'teamB_key': teamB['team_key'],
            'teamA_player1': teamA['player1'],
            'teamA_player2': teamA['player2'],
            'teamB_player1': teamB['player1'],
            'teamB_player2': teamB['player2'],

            # static context
            'teamA_avg_age': teamA['avg_age'],
            'teamB_avg_age': teamB['avg_age'],
            'age_diff': diff_or_nan(teamA['avg_age'], teamB['avg_age']),
            'teamA_avg_hgt': teamA['avg_hgt'],
            'teamB_avg_hgt': teamB['avg_hgt'],
            'hgt_diff': diff_or_nan(teamA['avg_hgt'], teamB['avg_hgt']),
            'teamA_rank': teamA['rank_main'],
            'teamB_rank': teamB['rank_main'],
            'rank_diff': diff_or_nan(teamA['rank_main'], teamB['rank_main']),
            'teamA_is_qualifier': teamA['is_qualifier'],
            'teamB_is_qualifier': teamB['is_qualifier'],
            'teamA_qualifier_num': teamA['qualifier_num'],
            'teamB_qualifier_num': teamB['qualifier_num'],
            'qualifier_num_diff': diff_or_nan(teamA['qualifier_num'], teamB['qualifier_num']),
            'one_side_qualifier': int(teamA['is_qualifier'] != teamB['is_qualifier']),
            'both_qualifier': int(teamA['is_qualifier'] == 1 and teamB['is_qualifier'] == 1),
            'teamA_same_country': teamA['same_country'],
            'teamB_same_country': teamB['same_country'],

            # pair/team history
            'teamA_is_new_pair': histA['is_new'],
            'teamB_is_new_pair': histB['is_new'],
            'teamA_prior_matches': histA['prior_matches'],
            'teamB_prior_matches': histB['prior_matches'],
            'pair_prior_matches_diff': diff_or_nan(histA['prior_matches'], histB['prior_matches']),
            'teamA_prior_win_pct': histA['prior_win_pct'],
            'teamB_prior_win_pct': histB['prior_win_pct'],
            'pair_prior_win_pct_diff': diff_or_nan(histA['prior_win_pct'], histB['prior_win_pct']),
            'teamA_recent3_win_pct': histA['recent3_win_pct'],
            'teamB_recent3_win_pct': histB['recent3_win_pct'],
            'recent3_win_pct_diff': diff_or_nan(histA['recent3_win_pct'], histB['recent3_win_pct']),
            'teamA_recent5_win_pct': histA['recent5_win_pct'],
            'teamB_recent5_win_pct': histB['recent5_win_pct'],
            'recent5_win_pct_diff': diff_or_nan(histA['recent5_win_pct'], histB['recent5_win_pct']),
            'teamA_recent10_win_pct': histA['recent10_win_pct'],
            'teamB_recent10_win_pct': histB['recent10_win_pct'],
            'recent10_win_pct_diff': diff_or_nan(histA['recent10_win_pct'], histB['recent10_win_pct']),
            'teamA_recent20_win_pct': histA['recent20_win_pct'],
            'teamB_recent20_win_pct': histB['recent20_win_pct'],
            'recent20_win_pct_diff': diff_or_nan(histA['recent20_win_pct'], histB['recent20_win_pct']),
            'teamA_last30_match_count': histA['last30_match_count'],
            'teamB_last30_match_count': histB['last30_match_count'],
            'last30_match_count_diff': diff_or_nan(histA['last30_match_count'], histB['last30_match_count']),
            'teamA_last30_win_pct': histA['last30_win_pct'],
            'teamB_last30_win_pct': histB['last30_win_pct'],
            'last30_win_pct_diff': diff_or_nan(histA['last30_win_pct'], histB['last30_win_pct']),
            'teamA_last90_match_count': histA['last90_match_count'],
            'teamB_last90_match_count': histB['last90_match_count'],
            'last90_match_count_diff': diff_or_nan(histA['last90_match_count'], histB['last90_match_count']),
            'teamA_last90_win_pct': histA['last90_win_pct'],
            'teamB_last90_win_pct': histB['last90_win_pct'],
            'last90_win_pct_diff': diff_or_nan(histA['last90_win_pct'], histB['last90_win_pct']),
            'teamA_last180_match_count': histA['last180_match_count'],
            'teamB_last180_match_count': histB['last180_match_count'],
            'last180_match_count_diff': diff_or_nan(histA['last180_match_count'], histB['last180_match_count']),
            'teamA_last180_win_pct': histA['last180_win_pct'],
            'teamB_last180_win_pct': histB['last180_win_pct'],
            'last180_win_pct_diff': diff_or_nan(histA['last180_win_pct'], histB['last180_win_pct']),
            'teamA_days_since_last_match': histA['days_since_last_match'],
            'teamB_days_since_last_match': histB['days_since_last_match'],
            'days_since_last_match_diff': diff_or_nan(histA['days_since_last_match'], histB['days_since_last_match']),

            # player summary histories
            'teamA_avg_player_prior_matches': playerA['avg_player_prior_matches'],
            'teamB_avg_player_prior_matches': playerB['avg_player_prior_matches'],
            'avg_player_prior_matches_diff': diff_or_nan(playerA['avg_player_prior_matches'], playerB['avg_player_prior_matches']),
            'teamA_avg_player_prior_win_pct': playerA['avg_player_prior_win_pct'],
            'teamB_avg_player_prior_win_pct': playerB['avg_player_prior_win_pct'],
            'avg_player_prior_win_pct_diff': diff_or_nan(playerA['avg_player_prior_win_pct'], playerB['avg_player_prior_win_pct']),
            'teamA_avg_player_recent10_win_pct': playerA['avg_player_recent10_win_pct'],
            'teamB_avg_player_recent10_win_pct': playerB['avg_player_recent10_win_pct'],
            'avg_player_recent10_win_pct_diff': diff_or_nan(playerA['avg_player_recent10_win_pct'], playerB['avg_player_recent10_win_pct']),
            'teamA_avg_player_recent20_win_pct': playerA['avg_player_recent20_win_pct'],
            'teamB_avg_player_recent20_win_pct': playerB['avg_player_recent20_win_pct'],
            'avg_player_recent20_win_pct_diff': diff_or_nan(playerA['avg_player_recent20_win_pct'], playerB['avg_player_recent20_win_pct']),
            'teamA_avg_player_last90_match_count': playerA['avg_player_last90_match_count'],
            'teamB_avg_player_last90_match_count': playerB['avg_player_last90_match_count'],
            'avg_player_last90_match_count_diff': diff_or_nan(playerA['avg_player_last90_match_count'], playerB['avg_player_last90_match_count']),
            'teamA_avg_player_last90_win_pct': playerA['avg_player_last90_win_pct'],
            'teamB_avg_player_last90_win_pct': playerB['avg_player_last90_win_pct'],
            'avg_player_last90_win_pct_diff': diff_or_nan(playerA['avg_player_last90_win_pct'], playerB['avg_player_last90_win_pct']),
            'teamA_avg_player_days_since_last_match': playerA['avg_player_days_since_last_match'],
            'teamB_avg_player_days_since_last_match': playerB['avg_player_days_since_last_match'],
            'avg_player_days_since_last_match_diff': diff_or_nan(playerA['avg_player_days_since_last_match'], playerB['avg_player_days_since_last_match']),
            'teamA_both_players_new': playerA['both_players_new'],
            'teamB_both_players_new': playerB['both_players_new'],

            # h2h
            'h2h_prior_matches': pair_hist['h2h_prior_matches'],
            'h2h_teamA_win_pct': h2h_a_win_pct,
            'h2h_recent3_teamA_win_pct': h2h_recent3_a_win_pct,

            # elo
            'teamA_elo': eloA,
            'teamB_elo': eloB,
            'elo_diff': eloA - eloB,
            'teamA_player1_elo': a_p1_elo,
            'teamA_player2_elo': a_p2_elo,
            'teamB_player1_elo': b_p1_elo,
            'teamB_player2_elo': b_p2_elo,
            'teamA_avg_player_elo': a_avg_player_elo,
            'teamB_avg_player_elo': b_avg_player_elo,
            'avg_player_elo_diff': a_avg_player_elo - b_avg_player_elo,
            'teamA_blend_elo': a_blend_elo,
            'teamB_blend_elo': b_blend_elo,
            'blend_elo_diff': a_blend_elo - b_blend_elo,

            'y': y,
        }
        rows.append(out)

        # update team history
        team_hist[teamA['team_key']].update(current_date, y)
        team_hist[teamB['team_key']].update(current_date, 1 - y)

        # update player history
        for key in [teamA['player1_key'], teamA['player2_key']]:
            player_hist[key].update(current_date, y)
        for key in [teamB['player1_key'], teamB['player2_key']]:
            player_hist[key].update(current_date, 1 - y)

        # update h2h
        h2h_hist[pair_key].update(y if teamA['team_key'] <= teamB['team_key'] else 1 - y)

        # update elos
        new_eloA, new_eloB = elo_update(eloA, eloB, y, k=k_factor)
        team_elo[teamA['team_key']] = new_eloA
        team_elo[teamB['team_key']] = new_eloB

        new_a_p1, new_b_p1 = elo_update(a_p1_elo, b_p1_elo, y, k=player_k_factor)
        new_a_p2, new_b_p2 = elo_update(a_p2_elo, b_p2_elo, y, k=player_k_factor)
        player_elo[teamA['player1_key']] = new_a_p1
        player_elo[teamA['player2_key']] = new_a_p2
        player_elo[teamB['player1_key']] = new_b_p1
        player_elo[teamB['player2_key']] = new_b_p2

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Build enriched prematch features for beach volleyball.')
    parser.add_argument('--input', required=True, help='Path to raw full_archive.csv')
    parser.add_argument('--output', required=True, help='Path to output prematch_features.csv')
    parser.add_argument('--k-factor', type=int, default=24, help='K factor for team Elo')
    parser.add_argument('--player-k-factor', type=int, default=20, help='K factor for player Elo')
    parser.add_argument('--blend-weight', type=float, default=0.5, help='Weight on team Elo in blended Elo')
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="latin-1", low_memory=False)
    built = build_dataset(df, k_factor=args.k_factor, player_k_factor=args.player_k_factor, blend_weight=args.blend_weight)
    built.to_csv(out, index=False)

    print(f'Input rows:  {len(df):,}')
    print(f'Output rows: {len(built):,}')
    print(f'Saved to:    {out}')


if __name__ == '__main__':
    main()

