from original_builder import *

def emit_row(row, team_hist, player_hist, h2h_hist, team_elo, player_elo, blend_weight=0.5):
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
    return out

def update_row(row, team_hist, player_hist, h2h_hist, team_elo, player_elo, k_factor=24, player_k_factor=20):
    current_date = row['date']
    w, l = get_team_dict(row, 'w'), get_team_dict(row, 'l')
    teamA, teamB = sorted([w, l], key=lambda t: t['team_key'])
    y = int(teamA['team_key'] == w['team_key'])
    pair_key = '||'.join(sorted([teamA['team_key'], teamB['team_key']]))
    eloA, eloB = team_elo[teamA['team_key']], team_elo[teamB['team_key']]
    a_p1_elo, a_p2_elo = player_elo[teamA['player1_key']], player_elo[teamA['player2_key']]
    b_p1_elo, b_p2_elo = player_elo[teamB['player1_key']], player_elo[teamB['player2_key']]

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

