import os, json, time

import globals.consts as g

import sample_collection.steamgames as sg
import sample_collection.steamcsv as sc
import mongodb.mongo as db


DATA_VALIDITY_SECONDS = g.DATA_VALIDITY_SECONDS

def get_all_basic_game_data():
    data = sg.get_game_data_trimmed()
    return data

def get_user_game_interactions():
    data = sc.get_interactions()
    return data

def create_name_to_appid_dict():
    data = db.get_data('name_to_appid')
    if data is not None:
        return data
    name_to_appid = _create_name_to_appid_dict()
    db.store_data('name_to_appid', name_to_appid)
    return name_to_appid

def _create_name_to_appid_dict():
    data = sg.get_game_data_trimmed()
    name_to_appid = {}
    for game in data.values():
        name_to_appid[game['name']] = game['appid']
    return name_to_appid

def get_appid(game_name):
    name_to_appid = create_name_to_appid_dict()
    if game_name not in name_to_appid:
        return 0
    return name_to_appid[game_name]

def get_games_in_csv():
    data = sc.get_games()
    return data