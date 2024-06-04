import os, json, requests, time
from tqdm import tqdm

import globals.consts as c

import mongodb.mongo as db

BASE_URL = 'https://api.curse.tools/v1/cf/'
GAME_LIST_URL = BASE_URL + 'games'
MOD_LIST_URL = BASE_URL + 'mods/search'

DATA_VALIDITY_SECONDS = c.DATA_VALIDITY_SECONDS

# Utility function to handle paginated requests
def unpaginate(url, params={}):
    results = []
    page = 1
    while True and page <= 200: # Limit to 10,000 results
        res = requests.get(url, params={**params, 'index': (page - 1) * 50, 'sortField': 2, 'sortOrder': 'desc'}) # Sort by popularity
        if res.status_code != 200:
            raise Exception('Failed to fetch ' + url + ' - ' + str(res.status_code) + ' - ' + res.text)
        data = res.json()
        if 'data' in data and len(data['data']) > 0:
            results.extend(data['data'])
        if data['pagination']['totalCount'] <= page * 50:
            break
        page += 1
    return results

def get_community_list():
    data = db.get_data('cf-communities')
    if data is not None:
        return data
    items = unpaginate(BASE_URL + 'games')
    db.store_data('cf-communities', items)
    return items

def get_identifiers():
    communities = get_community_list()
    identifiers = [x['id'] for x in communities]
    return identifiers

def get_packages(game_id):
    data = db.get_data('cf-packages-' + str(game_id))
    if data is not None:
        return data
    items = unpaginate(MOD_LIST_URL, params={'gameId': game_id})
    db.store_data('cf-packages-' + str(game_id), items)
    return items

def get_all_packages():
    identifiers = get_identifiers()
    packages = []
    for identifier in tqdm(identifiers, desc='Fetching CurseForge packages'):
        packages.extend(get_packages(identifier))
    return packages

def get_name_from_id(id):
    communities = get_community_list()
    for community in communities:
        if community['id'] == id:
            return community['name']
    return None
