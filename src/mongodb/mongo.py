'''
This script is used for storing data in a MongoDB database.
'''

import os, pymongo, json

import globals.consts as g

import tools.packages as pkg
import tools.tags as tgs


LOCAL_CACHE = g.LOCAL_CACHE


PASSWORD = os.environ.get('MONGO_PASSWORD')
FULL_URI = os.environ.get('MONGO_URI')
CONNECTION_STRING = 'mongodb://Ingo:' + PASSWORD + '@ingo.dog:27017' if PASSWORD is not None else 'mongodb://localhost:27017'
if FULL_URI is not None:
    CONNECTION_STRING = FULL_URI


client = None
db = None

def create_or_get_client():
    global client
    if client is None:
        client = pymongo.MongoClient(CONNECTION_STRING)
    return client

def create_or_get_db():
    global db
    if db is None:
        client = create_or_get_client()
        db = client['thesis']
    return db

def store_data(name, data):
    global db
    db = create_or_get_db()
    db[name].delete_many({})
    if type(data) == dict:
        data_formatted = {}
        for key in data:
            if data[key] == None or data[key] == {}:
                continue
            key_old = key
            if type(key) != str:
                key = str(key)
            d = data[key_old]
            if type(d) == list:
                d = {"_list": d}
            data_formatted[key] = d
            data_formatted[key]['_id'] = key
        if len(data_formatted) == 0:
            db[name].insert_one({})
        else:
            db[name].insert_many(data_formatted.values())
    else:
        if len(data) == 0:
            db[name].insert_one({})
        else:
            db[name].insert_many(data)
    if LOCAL_CACHE:
        if not os.path.exists('cache'):
            os.makedirs('cache')
        with open(os.path.join('cache', name + '.json'), 'w') as f:
            json.dump(clean(data), f, indent=4)

def get_data(name):
    if LOCAL_CACHE:
        if os.path.exists(os.path.join('cache', name + '.json')):
            with open(os.path.join('cache', name + '.json')) as f:
                return json.load(f)
    global db
    db = create_or_get_db()
    if name not in db.list_collection_names() or db[name].count_documents({}) == 0:
        return None
    data = clean(list(db[name].find()))
    if len(data) == 1 and data[0] == {}:
        if LOCAL_CACHE:
            if not os.path.exists('cache'):
                os.makedirs('cache')
            with open(os.path.join('cache', name + '.json'), 'w') as f:
                json.dump({}, f, indent=4)
        return []
    if LOCAL_CACHE:
        if not os.path.exists('cache'):
            os.makedirs('cache')
        with open(os.path.join('cache', name + '.json'), 'w') as f:
            json.dump(data, f, indent=4)
    return data

def clean(data):
    remove = []
    for item in data:
        if '_id' in item:
            if type(item['_id']) != str:
                item.pop('_id')
    for item in remove:
        if type(data) == dict:
            data.pop(item)
        else:
            data.remove(item)
    return data