import joblib
import os, random, json
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import globals.consts as g

import tools.packages as pkg
import tools.tags as tgs
import tools.games as gm

SEED = g.SEED

class Result:
    tag_name = ''
    value = None
    value2 = None
    value3 = None
    positive_count = 0
    negative_count = 0
    accuracy = 0
    positive_accuracy = 0
    negative_accuracy = 0

    def __init__(self, tag_name, positive_count, negative_count, accuracy, positive_accuracy, negative_accuracy, value=None, value2=None, value3=None):
        self.tag_name = tag_name
        self.positive_count = positive_count
        self.negative_count = negative_count
        self.accuracy = accuracy
        self.positive_accuracy = positive_accuracy
        self.negative_accuracy = negative_accuracy
        self.value = value
        self.value2 = value2
        self.value3 = value3
        
    def to_csv_string(self):
        if self.value3 != None:
            return f'{self.tag_name},{self.value},{self.value2},{self.value3},{self.positive_count},{self.negative_count},{self.accuracy},{self.positive_accuracy},{self.negative_accuracy}'
        elif self.value2 != None:
            return f'{self.tag_name},{self.value},{self.value2},{self.positive_count},{self.negative_count},{self.accuracy},{self.positive_accuracy},{self.negative_accuracy}'
        elif self.value != None:
            return f'{self.tag_name},{self.value},{self.positive_count},{self.negative_count},{self.accuracy},{self.positive_accuracy},{self.negative_accuracy}'
        else:
            return f'{self.tag_name},{self.positive_count},{self.negative_count},{self.accuracy},{self.positive_accuracy},{self.negative_accuracy}'

def get_x_y():
    packages = pkg.get_standardized_packages()

    X = []
    y = []
    for package in packages:
        desc = package['name'] + ' ' + package['description']
        if len(desc) < 50:
            continue
        X.append(desc)
        y.append(package['categories'])

    return X, y

def undersample(X, y, fallback_true, ratio):
    X_train = []
    y_train = []
    X_train_true = [x for i, x in enumerate(X) if y[i]]
    X_train_false = [x for i, x in enumerate(X) if not y[i]]

    if len(X_train_true) == 0:
        X_train_true = [fallback_true]

    random.seed(SEED)
    if int(len(X_train_true) * ratio) < len(X_train_false):
        X_train_false = random.sample(X_train_false, int(len(X_train_true) * ratio))

    X_train = X_train_true + X_train_false
    y_train = [True] * len(X_train_true) + [False] * len(X_train_false)

    return X_train, y_train

def prepare_train_test(X, y, fallback_true=None, do_undersampling=True, hard_limit=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    if do_undersampling:
        X_train, y_train = undersample(X_train, y_train, fallback_true, 1.5)

    if (y_test.count(True) == 0):
        X_test.append(fallback_true)
        y_test.append(True)
        
    X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

    if hard_limit is not None:
        X_train = X_train[:hard_limit]
        y_train = y_train[:hard_limit]

    return X_train, X_test, y_train, y_test
    

def save_models(models, name):
    if not os.path.exists('models'):
        os.makedirs('models')
    for tag in models:
        joblib.dump(models[tag], f'models/{name}-{esc(tag)}.joblib')

def save_results(results, name, value_name=None, value2_name=None, value3_name=None):
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(f'results/{name}.csv', 'w') as f:
        if value3_name is not None:
            f.write(f'Tag,{value_name},{value2_name},{value3_name},Positive Count,Negative Count,Accuracy,Positive Accuracy,Negative Accuracy\n')
        elif value2_name is not None:
            f.write(f'Tag,{value_name},{value2_name},Positive Count,Negative Count,Accuracy,Positive Accuracy,Negative Accuracy\n')
        elif value_name is not None:
            f.write(f'Tag,{value_name},Positive Count,Negative Count,Accuracy,Positive Accuracy,Negative Accuracy\n')
        else:
            f.write('Tag,Positive Count,Negative Count,Accuracy,Positive Accuracy,Negative Accuracy\n')
        for result in results:
            f.write(result.to_csv_string() + '\n')

def exponent_range(start, end):
    return [float(10 ** i) for i in range(start, end + 1)]

def smart_interpolate_range(start, end):
    if end is None:
        end = start
    step = (end - start) / 4
    begin = start - 2 * step
    arr = [begin + i * step for i in range(9)]
    arr = [x for x in arr if x > 0]
    return arr
    
INVALID_CHARS = ['/', '\\', '?', '*', ':', '|', '"', '<', '>']

def esc(tag):
    for char in INVALID_CHARS:
        tag = tag.replace(char, '')
    return tag

# Balanced accuracy for tags, since the dataset is imbalanced and negative tags appear much more frequently
def tag_accuracy(estimator, X, y):
    y_pred = estimator.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return weighted_accuracy(tn, fp, fn, tp)

def weighted_accuracy(tn, fp, fn, tp):
    total_negatives = tn + fp
    total_positives = fn + tp
    positive_accuracy = tp / total_positives
    negative_accuracy = tn / total_negatives
    return (positive_accuracy + negative_accuracy) / 2

def hyperparameter_tuning(model_builder, X_train, y_train, kf, initial_range, iterations):
    the_range = initial_range
    for _ in range(iterations):
        best_value = 1
        second_best_value = 1
        best_score = 0
        second_best_score = 0
        for value in the_range:
            model = model_builder(value)
            scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=tag_accuracy)
            score = scores.mean()
            if score > best_score:
                second_best_value = best_value
                second_best_score = best_score
                best_value = value
                best_score = score
            elif score > second_best_score:
                second_best_value = value
                second_best_score = score
        the_range = smart_interpolate_range(best_value, second_best_value)
    return best_value

def train_and_print_results(model, X_train, y_train, X_test, y_test, tag, value_name=None, value=None, value2_name=None, value2=None, value3_name=None, value3=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_negatives = tn + fp
    total_positives = fn + tp
    print(f'Tag: {tag}')
    if value_name is not None:
        print(f'{value_name}: {value}')
    if value2_name is not None:
        print(f'{value2_name}: {value2}')
    if value3_name is not None:
        print(f'{value3_name}: {value3}')
    print(f'Correct Negatives: {tn}/{total_negatives} ({tn/total_negatives:.2%})')
    print(f'Correct Positives: {tp}/{total_positives} ({tp/total_positives:.2%})')
    print(f'Tag Accuracy: {weighted_accuracy(tn, fp, fn, tp):.2%}')
    result = Result(tag, total_positives, total_negatives, weighted_accuracy(tn, fp, fn, tp), tp/total_positives, tn/total_negatives, value, value2, value3)
    return model, result

# Very basic model that checks if the description contains the tag
def baseline_description_contains():
    tags = tgs.get_classifier_tags()
    X, y_pre = get_x_y()
    results = []

    for tag in tags:
        y = [tag in tags for tags in y_pre]
        
        if y.count(True) == 0:
            print(f'No samples for tag {tag}, skipping')
            continue

        correct_positives = 0
        correct_negatives = 0
        total_positives = y.count(True)
        total_negatives = y.count(False)

        for i in range(len(y)):
            if y[i] == (tag.lower() in X[i].lower()):
                if y[i]:
                    correct_positives += 1
                else:
                    correct_negatives += 1

        print(f'Tag: {tag}')
        print(f'Correct Negatives: {correct_negatives}/{total_negatives} ({correct_negatives/total_negatives:.2%})')
        print(f'Correct Positives: {correct_positives}/{total_positives} ({correct_positives/total_positives:.2%})')
        print(f'Tag Accuracy: {weighted_accuracy(correct_negatives, total_negatives - correct_negatives, total_positives - correct_positives, correct_positives):.2%}')
        result = Result(tag, total_positives, total_negatives, weighted_accuracy(correct_negatives, total_negatives - correct_negatives, total_positives - correct_positives, correct_positives), correct_positives/total_positives, correct_negatives/total_negatives)
        results.append(result)

    save_results(results, 'baseline-description-contains')

# Train the Multinomial Naive Bayes model
# This model requires each tag to be trained separately, as it is a binary classifier
# The model is trained with hyperparameter tuning on alpha
def train_mnb(vectorizer=TfidfVectorizer(), vectorizer_name=''):
    tags = tgs.get_classifier_tags()
    X, y_pre = get_x_y()
    results = []

    models = {}
    for tag in tags:
        y = [tag in tags for tags in y_pre]
        
        if y.count(True) == 0:
            print(f'No samples for tag {tag}, skipping')
            continue

        fallback_true = X[(y.index(True))]

        X_train, X_test, y_train, y_test = prepare_train_test(X, y, fallback_true)

        if os.path.exists(f'models/mnb-{esc(tag)}{vectorizer_name}.joblib'):
            _model = joblib.load(f'models/mnb-{esc(tag)}{vectorizer_name}.joblib')
            best_alpha = _model.get_params()['multinomialnb__alpha']
        else:
            kf = KFold(n_splits=5)
            alphas = exponent_range(-4, 4)
            builder = lambda alpha: make_pipeline(vectorizer, MultinomialNB(alpha=alpha))
            best_alpha = hyperparameter_tuning(builder, X_train, y_train, kf, alphas, 5)
        
        model = make_pipeline(vectorizer, MultinomialNB(alpha=best_alpha))

        model, result = train_and_print_results(model, X_train, y_train, X_test, y_test, tag, 'Alpha', best_alpha)

        models[tag] = model
        results.append(result)

    save_models(models, 'mnb' + vectorizer_name)
    save_results(results, 'mnb' + vectorizer_name, 'Alpha')

# Train the BNB model
# (Same as MNB, but with Bernoulli Naive Bayes)
def train_bnb(vectorizer=TfidfVectorizer(), vectorizer_name=''):
    tags = tgs.get_classifier_tags()
    X, y_pre = get_x_y()
    results = []

    models = {}
    for tag in tags:
        y = [tag in tags for tags in y_pre]
        
        if y.count(True) == 0:
            print(f'No samples for tag {tag}, skipping')
            continue

        fallback_true = X[(y.index(True))]

        X_train, X_test, y_train, y_test = prepare_train_test(X, y, fallback_true)

        if os.path.exists(f'models/bnb-{esc(tag)}{vectorizer_name}.joblib'):
            _model = joblib.load(f'models/bnb-{esc(tag)}{vectorizer_name}.joblib')
            best_alpha = _model.get_params()['bernoullinb__alpha']
        else:
            kf = KFold(n_splits=5)
            alphas = exponent_range(-4, 4)
            builder = lambda alpha: make_pipeline(vectorizer, BernoulliNB(alpha=alpha))
            best_alpha = hyperparameter_tuning(builder, X_train, y_train, kf, alphas, 5)
        
        model = make_pipeline(vectorizer, BernoulliNB(alpha=best_alpha))

        model, result = train_and_print_results(model, X_train, y_train, X_test, y_test, tag, 'Alpha', best_alpha)

        models[tag] = model
        results.append(result)

    save_models(models, 'bnb' + vectorizer_name)
    save_results(results, 'bnb' + vectorizer_name, 'Alpha')

# Train the SVM model
# This model requires each tag to be trained separately, as it is a binary classifier
# The model is trained with hyperparameter tuning on C
def train_svm(kernel='linear', vectorizer=TfidfVectorizer(), vectorizer_name=''):
    tags = tgs.get_classifier_tags()
    X, y_pre = get_x_y()
    results = []

    models = {}
    for tag in tags:
        y = [tag in tags for tags in y_pre]
        
        if y.count(True) == 0:
            print(f'No samples for tag {tag}, skipping')
            continue

        fallback_true = X[(y.index(True))]

        X_train, X_test, y_train, y_test = prepare_train_test(X, y, fallback_true, True, 1000)

        if (os.path.exists(f'models/svm-{esc(tag)}{vectorizer_name}.joblib') and kernel == 'linear') or (os.path.exists(f'models/svm-{kernel}-{esc(tag)}{vectorizer_name}.joblib') and kernel != 'linear'):
            _model = joblib.load(f'models/svm-{esc(tag)}{vectorizer_name}.joblib') if kernel == 'linear' else joblib.load(f'models/svm-{kernel}-{esc(tag)}{vectorizer_name}.joblib')
            best_C = _model.get_params()['svc__C']
        else:
            kf = KFold(n_splits=5)
            Cs = exponent_range(-4, 4)
            builder = lambda C: make_pipeline(vectorizer, SVC(C=C, kernel=kernel))
            best_C = hyperparameter_tuning(builder, X_train, y_train, kf, Cs, 5)
        
        model = make_pipeline(vectorizer, SVC(C=best_C, kernel=kernel))

        model, result = train_and_print_results(model, X_train, y_train, X_test, y_test, tag, 'C', best_C)

        models[tag] = model
        results.append(result)

    save_models(models, f'svm{vectorizer_name}' if kernel == 'linear' else f'svm-{kernel}{vectorizer_name}')
    save_results(results, f'svm{vectorizer_name}' if kernel == 'linear' else f'svm-{kernel}{vectorizer_name}', 'C')

# Train the Logistic Regression model
# This model requires each tag to be trained separately, as it is a binary classifier
# The model is trained with hyperparameter tuning on C
def train_logistic_regression(vectorizer=TfidfVectorizer(), vectorizer_name=''):
    tags = tgs.get_classifier_tags()
    X, y_pre = get_x_y()
    results = []

    models = {}
    for tag in tags:
        y = [tag in tags for tags in y_pre]
        
        if y.count(True) == 0:
            print(f'No samples for tag {tag}, skipping')
            continue

        fallback_true = X[(y.index(True))]

        X_train, X_test, y_train, y_test = prepare_train_test(X, y, fallback_true, True)

        if os.path.exists(f'models/logistic-regression-{esc(tag)}{vectorizer_name}.joblib'):
            _model = joblib.load(f'models/logistic-regression-{esc(tag)}{vectorizer_name}.joblib')
            best_C = _model.get_params()['logisticregression__C']
        else:
            kf = KFold(n_splits=5)
            Cs = exponent_range(-4, 4)
            builder = lambda C: make_pipeline(vectorizer, LogisticRegression(C=C))
            best_C = hyperparameter_tuning(builder, X_train, y_train, kf, Cs, 5)
        
        model = make_pipeline(vectorizer, LogisticRegression(C=best_C))

        model, result = train_and_print_results(model, X_train, y_train, X_test, y_test, tag, 'C', best_C)

        models[tag] = model
        results.append(result)

    save_models(models, 'logistic-regression' + vectorizer_name)
    save_results(results, 'logistic-regression' + vectorizer_name, 'C')

def classify_all_games():
    tags = tgs.get_classifier_tags()
    model_dict_mnb = {tag: joblib.load(f'models/mnb-{esc(tag)}.joblib') for tag in tqdm(tags, desc='Loading MNB models')}
    model_dict_svm = {tag: joblib.load(f'models/svm-{esc(tag)}.joblib') for tag in tqdm(tags, desc='Loading SVM models')}
    model_dict_lr = {tag: joblib.load(f'models/logistic-regression-{esc(tag)}.joblib') for tag in tqdm(tags, desc='Loading LR models')}
    games = gm.get_games_in_csv()
    data = gm.get_all_basic_game_data()
    descs = {game['name']: game['description'] for game in data.values()}
    generated_tags = {}
    for game in games:
        if not game in descs:
            continue
        desc = game + ' ' + descs[game]
        predictions_mnb = {tag: model_dict_mnb[tag].predict([desc])[0] for tag in tags}
        predictions_svm = {tag: model_dict_svm[tag].predict([desc])[0] for tag in tags}
        predictions_lr = {tag: model_dict_lr[tag].predict([desc])[0] for tag in tags}
        # If at least 2/3 models agree on a tag being present (True), add it to the game
        tags_combined = [tag for tag in tags if sum([predictions_mnb[tag], predictions_svm[tag], predictions_lr[tag]]) >= 2]
        generated_tags[game] = tags_combined

    with open('generated_tags.json', 'w') as f:
        json.dump(generated_tags, f, indent=4)