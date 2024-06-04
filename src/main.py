import os, re

import classification.classification_models as clf
import recommendation.recommendation_models as rec
import tools.packages as pkg
import tools.tags as tags
from sklearn.feature_extraction.text import TfidfVectorizer

from prompt_toolkit import prompt, completion

custom_vectorizer = TfidfVectorizer()
vectorizer_name = ''

k = 10

class Command:
    def __init__(self, name, description, function=None):
        self.name = name
        self.description = description
        self.function = function

def do_help():
    print('Commands:')
    cmds = [command for command in commands]
    cmds.sort(key=lambda x: x.name)
    for command in cmds:
        print(f'  {command.name}: {command.description}')

def do_package_info():
    print('Getting package information...')
    pkg.print_info()
    
def do_baseline_description_contains():
    print('Getting the baseline results...')
    clf.baseline_description_contains()

def do_train_mnb():
    print('Training the Multinomial Naive Bayes model...')
    clf.train_mnb(vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_bnb():
    print('Training the Bernoulli Naive Bayes model...')
    clf.train_bnb(vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_svm():
    print('Training the Support Vector Machine model...')
    clf.train_svm(vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_svm_rbf():
    print('Training the SVM model with the RBF kernel...')
    clf.train_svm(kernel='rbf', vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_svm_poly():
    print('Training the SVM model with the polynomial kernel...')
    clf.train_svm(kernel='poly', vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_svm_sigmoid():
    print('Training the SVM model with the sigmoid kernel...')
    clf.train_svm(kernel='sigmoid', vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_lr():
    print('Training the Logistic Regression model...')
    clf.train_logistic_regression(vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_all_cls():
    print('Training all classification models...')
    do_train_mnb()
    do_train_svm()
    do_train_lr()
    
def do_train_knn():
    print('Training the K-Nearest Neighbors model...')
    rec.train_knn(k, vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_knn_desc_only():
    print('Training the K-Nearest Neighbors model with only the description...')
    rec.train_knn(k, 'description', vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_knn_tags_only():
    print('Training the K-Nearest Neighbors model with only the tags...')
    rec.train_knn(k, 'tags', vectorizer=custom_vectorizer, vectorizer_name=vectorizer_name)

def do_train_knn_collab():
    print('Training the collaborative filtering K-Nearest Neighbors model...')
    rec.train_knn_collab(k)

def do_train_all_rec():
    print('Training all recommendation models...')
    do_train_knn()
    do_train_knn_desc_only()
    do_train_knn_tags_only()
    do_train_knn_collab()

def set_k(value):
    global k
    k = value

def do_set_vectorizer_tfidf():
    print('Setting the vectorizer to TF-IDF...')
    global custom_vectorizer, vectorizer_name
    custom_vectorizer = TfidfVectorizer()
    vectorizer_name = ''

def do_set_vectorizer_tfidf_stopwords():
    print('Setting the vectorizer to TF-IDF with stopwords...')
    global custom_vectorizer, vectorizer_name
    custom_vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer_name = '-stopwords'

def do_set_vectorizer_tfidf_ngrams():
    print('Setting the vectorizer to TF-IDF with n-grams...')
    global custom_vectorizer, vectorizer_name
    custom_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer_name = '-ngrams'

def do_set_vectorizer_tfidf_stopwords_and_ngrams():
    print('Setting the vectorizer to TF-IDF with stopwords and n-grams...')
    global custom_vectorizer, vectorizer_name
    custom_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectorizer_name = '-stopwords-ngrams'

def do_generate_ugc_tag_counts():
    print('Generating tag counts for all UGC packages and saving them to a file...')
    tags.generate_ugc_tag_counts()

def do_generate_game_tag_counts():
    print('Generating tag counts for all games and saving them to a file...')
    tags.generate_game_tag_counts()

def do_exit():
    print('I thought we were friends...')
    exit()
    
commands = [
    # Info commands
    Command('help', 'Show this help message', do_help),
    # Package info commands
    Command('packageinfo', 'Get information about the packages', do_package_info),
    # Classification commands
    Command('trainc baseline', 'Get the baseline results (Not really training)', do_baseline_description_contains),
    Command('trainc mnb', 'Train the Multinomial Naive Bayes model', do_train_mnb),
    Command('trainc bnb', 'Train the Bernoulli Naive Bayes model', do_train_bnb),
    Command('trainc svm', 'Train the Support Vector Machine model', do_train_svm),
    Command('trainc svm linear', 'Same as "trainc svm"', do_train_svm),
    Command('trainc svm rbf', 'Train the SVM model with the RBF kernel', do_train_svm_rbf),
    Command('trainc svm poly', 'Train the SVM model with the polynomial kernel', do_train_svm_poly),
    Command('trainc svm sigmoid', 'Train the SVM model with the sigmoid kernel', do_train_svm_sigmoid),
    Command('trainc lr', 'Train the Logistic Regression model', do_train_lr),
    Command('trainc all', 'Train all classification models', do_train_all_cls),
    # Recommendation commands
    Command('trainr knn', 'Train the K-Nearest Neighbors model', do_train_knn),
    Command('trainr knn_desc_only', 'Train the K-Nearest Neighbors model with only the description', do_train_knn_desc_only),
    Command('trainr knn_tags_only', 'Train the K-Nearest Neighbors model with only the tags', do_train_knn_tags_only),
    Command('trainr knn_collab', 'Train the collaborative filtering K-Nearest Neighbors model', do_train_knn_collab),
    Command('trainr all', 'Train all recommendation models', do_train_all_rec),
    # Options commands
    Command('vectorizer tfidf', 'Use the TF-IDF vectorizer', do_set_vectorizer_tfidf),
    Command('vectorizer tfidf-stopwords', 'Use the TF-IDF vectorizer with stopwords', do_set_vectorizer_tfidf_stopwords),
    Command('vectorizer tfidf-ngrams', 'Use the TF-IDF vectorizer with n-grams', do_set_vectorizer_tfidf_ngrams),
    Command('vectorizer tfidf-stopwords-ngrams', 'Use the TF-IDF vectorizer with stopwords and n-grams', do_set_vectorizer_tfidf_stopwords_and_ngrams),
    Command('setk', 'Set the number of neighbors for KNN', lambda: set_k(int(prompt('Enter the value of k: ')))),
    # Tag commands
    Command('tagcount ugc', 'Generate tag counts for all UGC packages and save them to a file', do_generate_ugc_tag_counts),
    Command('tagcount games', 'Generate tag counts for all games and save them to a file', do_generate_game_tag_counts),
    # Utility commands
    Command('multi', 'Run multiple commands'),
    Command('custom', 'Run a function manually'),
    # Exit command
    Command('exit', 'Exit the program', do_exit)
]

from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style


command_starts = ['help', 'packageinfo', 'trainc', 'trainr', 'vectorizer', 'setk', 'tagcount', 'multi', 'custom', 'exit']

cls_names = ['baseline', 'mnb', 'bnb', 'svm', 'lr', 'all']

cls_svm_subnames = ['linear', 'rbf', 'poly', 'sigmoid']

rec_names = ['knn', 'knn_desc_only', 'knn_tags_only', 'knn_collab', 'all']

vec_names = ['tfidf', 'tfidf-stopwords', 'tfidf-ngrams', 'tfidf-stopwords-ngrams']

tag_names = ['ugc', 'games']

multi_on = False

class CustomLexer(Lexer):
    def lex_document(self, document):
        global multi_on
        text = document.text

        tokens = re.split(r'( +)', text)
        if tokens[0] == '':
            tokens = tokens[1:]
        if len(tokens) > 0 and tokens[-1] == '':
            tokens = tokens[:-1]

        def get_style(line, text, cmds):
            if line == 0:
                if multi_on and text == 'multi':
                    return 'class:default'
                if text in command_starts:
                    return 'class:param1'
                if multi_on and text == 'undo' or text == 'cancel':
                    return 'class:param1'
            if line == 2:
                if cmds[0] == 'trainc' and text in cls_names:
                    return 'class:param2'
                if cmds[0] == 'trainr' and text in rec_names:
                    return 'class:param2'
                if cmds[0] == 'vectorizer' and text in vec_names:
                    return 'class:param2'
                if cmds[0] == 'tagcount' and text in tag_names:
                    return 'class:param2'
            if line == 4:
                if cmds[0] == 'trainc' and cmds[2] == 'svm' and text in cls_svm_subnames:
                    return 'class:param3'
            return 'class:default'

        def lexer_function(_):
            global base_commands
            tuples = []
            for i in range(len(tokens)):
                text = tokens[i]
                tuples.append((get_style(i, text, tokens), text))
            return tuples
                    
        return lexer_function

custom_style = Style.from_dict({
    'param1': 'green',
    'param2': 'cyan',
    'param3': 'yellow',
})
    

print('Please type the command you want to run.')
print('Type "help" for a list of commands. Press the Tab key for autocompletion.')
print('Type "exit" to exit the program.')
l = CustomLexer()
while True:
    cmds = [command.name for command in commands]
    cmds.sort()
    text = prompt('>>> ',
        completer=completion.WordCompleter(cmds, ignore_case=True, sentence=True),
        complete_while_typing=True,
        lexer=l,
        style=custom_style)
    if text.lower() == 'multi':
        multi_on = True
        print('Multi-command mode enabled. Please type the commands (separated by Enter)')
        print('Leave empty to run the commands. Type "cancel" to cancel. Type "undo" to remove the previously entered command.')
        command_list = []
        while True:
            multicommands = [command.name for command in commands]
            multicommands.append('undo')
            multicommands.append('cancel')
            multicommands.remove('multi')
            multicommands.sort()
            text = prompt('MM> ',
                completer=completion.WordCompleter(multicommands, ignore_case=True, sentence=True),
                complete_while_typing=True,
                lexer=l,
                style=custom_style)
            if text.lower() == 'cancel':
                print('Multi-command mode disabled.')
                break
            elif text.lower() == 'help':
                do_help()
            elif text.lower() == 'exit':
                do_exit()
            elif text.lower() == 'multi':
                print('Multi-command mode is already enabled. Type "cancel" to disable it.')
            elif text.lower() == 'undo':
                if len(command_list) > 0:
                    command_list.pop()
                else:
                    print('No commands to undo.')
            elif text == '':
                if len(command_list) == 0:
                    print('No commands entered. Exiting multi-command mode.')
                    break
                print('Running commands...')
                for command in command_list:
                    command.function()
                print('All commands have been run.')
                break
            else:
                for command in commands:
                    if text.lower() == command.name:
                        command_list.append(command)
                        break
                else:
                    print('Command not found. Type "help" for a list of commands.')
                    continue
        multi_on = False
    elif text.lower() == 'custom':
        py_files = []
        for root, _, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
        functions = []
        for file in py_files:
            with open(file, 'r') as f:
                lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith('def '):
                    function_name = line.split(' ')[1].split('(')[0]
                    f = file[4:-3].replace('/', '.').replace('\\', '.')
                    functions.append(f'{f}::{function_name}')
        functions.sort()
        text = prompt('Select a function to run:\n', completer=completion.WordCompleter(functions, ignore_case=True, sentence=True))
        try:
            parts = text.split('::')
            imp = __import__(parts[0], fromlist=[parts[1]])
            getattr(imp, parts[1])()
        except Exception as e:
            print(f'Error running function: {e}')
    else:
        for command in commands:
            if text.lower() == command.name:
                command.function()
                break
        else:
            print('Command not found. Type "help" for a list of commands.')