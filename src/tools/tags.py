import json, os
from fuzzywuzzy import fuzz
from tqdm import tqdm

import tools.packages as pkg
import tools.games as games

def generate_ugc_tag_counts():
    packages = pkg.standardize_all()
    tag_counts = []
    for package in tqdm(packages, desc='Generating tag counts'):
        for tag in package['categories']:
            if tag == '' or tag == None:
                continue
            found = False
            for tag_count in tag_counts:
                if tag_count['_id'] == tag:
                    tag_count['count'] += 1
                    found = True
                    break
            if not found:
                tag_counts.append({'_id': tag, 'count': 1})
    tag_counts = sorted(tag_counts, key=lambda x: x['count'], reverse=True)
    with open('ugc_tag_counts.json', 'w') as f:
        json.dump(tag_counts, f, indent=4)

def generate_standardization_data():
    with open('ugc_tag_counts.json', 'r') as f:
        tag_counts = json.load(f)
    tag_standardize_dict = {}

    for tag in tqdm(tag_counts, desc='Generating tag standardization data'):
        best_match = None
        best_match_similarity = 90
        for tag2 in tag_counts:
            if tag == tag2:
                continue
            similarity = fuzz.ratio(tag['_id'].lower(), tag2['_id'].lower())
            match = False
            if similarity >= best_match_similarity:
                match = True
            if tag2['_id'].lower() == tag['_id'].lower() + 's' or tag['_id'].lower() == tag2['_id'].lower() + 's' or tag2['_id'].lower() == tag['_id'].lower():
                match = True
                similarity = 100
            if match and similarity > best_match_similarity and tag2['count'] > tag['count'] and tag2['count'] >= 100:
                best_match = tag2['_id']
                best_match_similarity = similarity
        if best_match is not None:
            tag_standardize_dict[tag['_id']] = best_match
        else:
            tag_standardize_dict[tag['_id']] = tag['_id'] if tag['count'] >= 100 else None

    with open('ugc_tag_standardize_dict.json', 'w') as f:
        json.dump(tag_standardize_dict, f, indent=4)

def standardize(array):
    if not os.path.exists('ugc_tag_standardize_dict.json'):
        generate_standardization_data()
    with open('ugc_tag_standardize_dict.json', 'r') as f:
        tag_standardize_dict = json.load(f)
    for i in range(len(array)):
        if array[i] in tag_standardize_dict:
            array[i] = tag_standardize_dict[array[i]]
    return array

def get_classifier_tags():
    with open('tags.txt', 'r') as f:
        tags = f.readlines()
    tags = [x.strip() for x in tags]
    return tags

def generate_game_tag_counts():
    data = games.get_all_basic_game_data()
    tag_counts = []
    for game in tqdm(data, desc='Generating game tag counts'):
        for tag in data[game]['tags']:
            if tag == '' or tag == None:
                continue
            found = False
            for tag_count in tag_counts:
                if tag_count['_id'] == tag:
                    tag_count['count'] += 1
                    found = True
                    break
            if not found:
                tag_counts.append({'_id': tag, 'count': 1})
    tag_counts = sorted(tag_counts, key=lambda x: x['count'], reverse=True)
    with open('game_tag_counts.json', 'w') as f:
        json.dump(tag_counts, f, indent=4)
