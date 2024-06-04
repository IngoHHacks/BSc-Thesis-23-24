import os, json, requests, time
from tqdm import tqdm

import globals.consts as c

import mongodb.mongo as db


BASE_URL = 'https://thunderstore.io/'
COMMUNITY_LIST_URL = BASE_URL + 'api/experimental/community/'
PACKAGE_LIST_URL = BASE_URL + 'c/%s/api/v1/package/'

DATA_VALIDITY_SECONDS = c.DATA_VALIDITY_SECONDS

def get_community_list():
    data = db.get_data('thunderstore-communities')
    if data is not None:
        return data
    items = requests.get(COMMUNITY_LIST_URL).json()['results']
    db.store_data('thunderstore-communities', items)
    return items

def get_identifiers():
    communities = get_community_list()
    identifiers = [x['identifier'] for x in communities]
    return identifiers

def get_packages(identifier):
    data = db.get_data('thunderstore-packages-' + identifier)
    if data is not None:
        return data
    items = requests.get(PACKAGE_LIST_URL % identifier).json()
    db.store_data('thunderstore-packages-' + identifier, items)
    return items

def get_all_packages():
    identifiers = get_identifiers()
    packages = []
    for identifier in tqdm(identifiers, desc='Fetching Thunderstore packages'):
        cur = get_packages(identifier)
        for package in cur:
            package['community_identifier'] = identifier
        packages.extend(cur)
    return packages


        