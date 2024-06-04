import os, json, requests, time
from tqdm import tqdm

import globals.consts as c

import sample_collection.steamcsv as steamcsv
import mongodb.mongo as db

from fuzzywuzzy import fuzz

STEAM_WEB_API_KEY = c.STEAM_WEB_API_KEY

DATA_VALIDITY_SECONDS = c.DATA_VALIDITY_SECONDS

QUERY_GAMES = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"

def query_games():
    data = db.get_data('steam-games')
    if data:
        return data
    list = _query_games()
    db.store_data('steam-games', list)
    return list

def _query_games():
    games = []
    url = QUERY_GAMES
    response = requests.get(url)
    json_response = response.json()
    games.extend(json_response['applist']['apps'])
    return games
    
def get_game_data():
    data = db.get_data('steam-game-data')
    if data is not None:
        if type(data) == list:
            data = c.l2d(data)
        return data
    data = _get_game_data()
    db.store_data('steam-game-data', data)
    return data
    
def _get_game_data():
    game_list = steamcsv.get_games()
    json_data = {}
    for game in tqdm(game_list, desc='Fetching Steam game data'):
        # Lookup the game by name using the Steam Web API (get the appid)
        url = f"https://steamcommunity.com/actions/SearchApps/${game}"
        response = requests.get(url)
        wt = 1
        while response.status_code != 200:
            time.sleep(wt)
            wt *= 2
            response = requests.get(url)
        json_response = response.json()
        appid = json_response[0]['appid'] if json_response and json_response[0] and 'appid' in json_response[0] else None
        if appid is None:
            continue
        json_data[game] = get_game_data_for_id(appid)
    return json_data

def get_game_data_for_id(id):
    url = f"https://store.steampowered.com/api/appdetails?appids={id}"
    response = requests.get(url)
    wt = 1
    while response.status_code != 200:
        time.sleep(wt)
        wt *= 2
        response = requests.get(url)
    json_response = response.json()
    if json_response[f'{id}']['success']:
        return json_response[f'{id}']['data']
    return None
        
def get_game_data_trimmed():
    data = db.get_data('steam-game-data-trimmed')
    if data is not None:
        if type(data) == list:
            data = c.l2d(data)
        return data
    game_data = get_game_data()
    trimmed_game_data = {}
    for game in game_data:
        if game_data[game] == None or not 'name' in game_data[game] or not 'steam_appid' in game_data[game] or not 'short_description' in game_data[game] or not 'genres' in game_data[game] or not 'categories' in game_data[game]:
            continue
        trimmed_game_data[game] = {}
        trimmed_game_data[game]['name'] = game_data[game]['name']
        trimmed_game_data[game]['appid'] = game_data[game]['steam_appid']
        trimmed_game_data[game]['description'] = game_data[game]['short_description']
        trimmed_game_data[game]['tags'] = [x['description'] for x in game_data[game]['genres']] + [x['description'] for x in game_data[game]['categories']]
    db.store_data('steam-game-data-trimmed', trimmed_game_data)
    return trimmed_game_data