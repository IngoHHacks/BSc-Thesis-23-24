import os, json, time
from pandas import read_csv

import globals.consts as c

import mongodb.mongo as db


DATA_VALIDITY_SECONDS = c.DATA_VALIDITY_SECONDS

# Ignore users with less than 10 'play' interactions
def get_interactions_filtered():
    interactions = get_interactions()
    interactions_filtered = {}
    for user_id in interactions:
        play_count = 0
        for interaction in interactions[user_id]:
            if interaction['interaction'] == 'play':
                play_count += 1
        if play_count >= 10:
            interactions_filtered[user_id] = interactions[user_id]
    return interactions_filtered

def get_interactions():
    data = db.get_data('steam-interactions')
    if data is not None:
        if type(data) == list:
            data = c.l2d(data)
        data_formatted = {}
        for user_id in data:
            data_formatted[int(user_id)] = data[user_id]
        return data_formatted
    interactions = _get_interactions()
    db.store_data('steam-interactions', interactions)
    return interactions


def _get_interactions():
    steam = read_csv('steam-200k.csv', header=None)

    interactions = {}
    for _, row in steam.iterrows():
        user_id = row[0]
        game_name = row[1]
        interaction = row[2]
        hours_played = row[3] if interaction == 'play' else -1
        if user_id not in interactions:
            interactions[user_id] = []
        interactions[user_id].append({'game_name': game_name, 'interaction': interaction, 'hours_played': hours_played})
    return interactions

def get_games():
    steam = read_csv('steam-200k.csv', header=None)
    return steam[1].unique()