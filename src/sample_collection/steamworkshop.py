import os, json, requests, time, urllib
from tqdm import tqdm

import globals.consts as c

import mongodb.mongo as db

STEAM_WEB_API_KEY = c.STEAM_WEB_API_KEY

DATA_VALIDITY_SECONDS = c.DATA_VALIDITY_SECONDS

QUERY_FILES = "https://api.steampowered.com/IPublishedFileService/QueryFiles/v1/?key=" + STEAM_WEB_API_KEY + "&query_type=%i&cursor=%s&numperpage=%i&filetype=%i&return_tags=true&return_vote_data=true&return_metadata=true"

def query_workshop_items(numperpage = 100, limit=100000, cursor = "*"):
    data = db.get_data('steam-workshop-items')
    if data is not None:
        print('Fetching Steam Workshop items from cache')
        return data
    items = _query_workshop_items(numperpage, limit, cursor)
    db.store_data('steam-workshop-items', items)
    return items

def _query_workshop_items(numperpage = 100, limit=100000, cursor = "*"):
    items = []
    url = QUERY_FILES % (11, urllib.parse.quote(cursor), numperpage, 18) # query_type 11 is for ranked by popularity, filetype 18 is for ready-to-use Steam Workshop items
    response = requests.get(url)
    json_response = response.json()
    total = json_response['response']['total']
    if total > limit:
        total = limit
    for i in tqdm(range(0, total, numperpage), desc='Fetching Steam Workshop items'):
        url = QUERY_FILES % (11, urllib.parse.quote(cursor), numperpage, 18)
        response = requests.get(url)
        json_response = response.json()
        if json_response['response'].get('publishedfiledetails') is None:
            break
        published_file_details = json_response['response']['publishedfiledetails']
        items.extend(published_file_details)
        cursor = json_response['response']['next_cursor']
    return items