import os, json, time

import globals.consts as c

import sample_collection.thunderstore as ts
import sample_collection.curseforge as cf
import sample_collection.steamworkshop as sw
import tools.tags as tgs
import mongodb.mongo as db

DATA_VALIDITY_SECONDS = c.DATA_VALIDITY_SECONDS

def standardize_ts(packages):
    standardized = []
    seen_ids = set()
    for package in packages:
        if package['has_nsfw_content'] or package['is_deprecated']: # Skip NSFW and deprecated packages
            continue
        _id = 'ts/' + package['community_identifier'] + '/' + package['uuid4']
        if _id in seen_ids: # Skip duplicates
            continue
        new_package = {}
        new_package['_id'] = _id
        new_package['source'] = 'Thunderstore'
        new_package['name'] = package['full_name']
        new_package['description'] = package['versions'][0]['description']
        new_package['icon'] = package['versions'][0]['icon']
        new_package['community'] = package['community_identifier']
        new_package['creators'] = [package['owner']]
        new_package['categories'] = package['categories']
        new_package['downloads'] = sum([x['downloads'] for x in package['versions']])
        new_package['likes'] = package['rating_score']
        standardized.append(new_package)
        seen_ids.add(_id)
    return standardized

def standardize_cf(packages):
    standardized = []
    seen_ids = set()
    for package in packages:
        if package['status'] != 4 or not package['isAvailable']: # Skip unapproved and unavailable packages
            continue
        _id =  'cf/' + str(package['id'])
        if _id in seen_ids:
            continue
        new_package = {}
        new_package['_id'] = _id
        new_package['source'] = 'CurseForge'
        new_package['name'] = package['name']
        new_package['description'] = package['summary']
        new_package['icon'] = package['logo']['thumbnailUrl'] if package['logo'] != None else 'https://ingoh.net/images/placeholder_curseforge.png'
        new_package['community'] = cf.get_name_from_id(package['gameId'])
        new_package['creators'] = [x['name'] for x in package['authors']]
        new_package['categories'] = [x['name'] for x in package['categories']]
        new_package['downloads'] = package['downloadCount']
        new_package['likes'] = package['thumbsUpCount']
        standardized.append(new_package)
        seen_ids.add(_id)
    return standardized

def standardize_sw(packages):
    standardized = []
    seen_ids = set()
    for package in packages:
        if package['result'] != 1 or package['visibility'] != 0 or package['banned'] or ('maybe_inappropriate_sex' in package and package['maybe_inappropriate_sex']): # Skip NSFW, banned, and unlisted packages
            continue
        _id = 'sw/' + str(package['publishedfileid'])
        if _id in seen_ids:
            continue
        new_package = {}
        new_package['_id'] = _id
        new_package['source'] = 'SteamWorkshop'
        new_package['name'] = package['title']
        new_package['description'] = package['file_description']
        new_package['icon'] = package['preview_url']
        new_package['community'] = package['app_name']
        new_package['creators'] = [package['creator']]
        new_package['categories'] = [x['tag'] for x in package['tags']] if 'tags' in package else []
        new_package['downloads'] = package['subscriptions'] if 'subscriptions' in package else 0
        new_package['likes'] = package['favorited'] if 'favorited' in package else 0
        standardized.append(new_package)
        seen_ids.add(_id)
    return standardized

def standardize_all():
    data = db.get_data('all-packages')
    if data is not None:
        return data
    print('Standardizing packages... (This may take a while)')
    ts_packages = ts.get_all_packages()
    cf_packages = cf.get_all_packages()
    sw_packages = sw.query_workshop_items()
    all_packages = standardize_ts(ts_packages) + standardize_cf(cf_packages) + standardize_sw(sw_packages)
    for package in all_packages:
        package['categories'] = tgs.standardize(package['categories'])
    db.store_data('all-packages', all_packages)
    return all_packages

def get_standardized_packages():
    return standardize_all()

def print_info():
    print('Package information:')
    t_items = standardize_ts(ts.get_all_packages())
    print('Total Thunderstore packages:', len(t_items))
    c_items = standardize_cf(cf.get_all_packages())
    print('Total CurseForge packages:', len(c_items))
    w_items = standardize_sw(sw.query_workshop_items())
    print('Total Steam Workshop packages:', len(w_items))
    all_items = t_items + c_items + w_items
    print('Total packages:', len(all_items))
    t_communities = list(set([x['community'] for x in t_items]))
    print('Total Thunderstore communities:', len(t_communities))
    c_communities = list(set([x['community'] for x in c_items]))
    print('Total CurseForge communities:', len(c_communities))
    w_communities = list(set([x['community'] for x in w_items]))
    print('Total Steam Workshop communities:', len(w_communities))
    print('Total communities:', len(t_communities + c_communities + w_communities))
    tags = list(set([x for y in [x['categories'] for x in all_items] for x in y]))
    print('Total tags:', len(tags))