import os

STEAM_WEB_API_KEY = os.environ.get('STEAM_WEB_API_KEY')
DATA_VALIDITY_SECONDS = 1e9999 # Keep data indefinitely since it updates infrequently. (Data can be manually refreshed by deleting the data file in the MongoDB database)
LOCAL_CACHE = True
SEED = 1

def l2d(l):
    d = {}
    for item in l:
        item_without_id = item.copy()
        item_without_id.pop('_id')
        if '_list' in item_without_id:
            item_without_id = item_without_id['_list']
        d[item['_id']] = item_without_id
    return d