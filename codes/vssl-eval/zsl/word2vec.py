
""" this recipe is from 
https://github.com/bbrattoli/ZeroShotVideoClassification/
"""

import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors as Word2Vec
from sklearn.preprocessing import normalize

def classes2embedding(dataset_name, class_name_inputs, wv_model):
    if dataset_name == 'ucf101':
        one_class2embed = one_class2embed_ucf
    elif dataset_name == 'hmdb51':
        one_class2embed = one_class2embed_hmdb
    elif dataset_name == 'kinetics400':
        one_class2embed = one_class2embed_kinetics400
    elif dataset_name == 'rareact':
        one_class2embed = one_class2embed_rareact
    elif dataset_name == 'kinetics700':
        one_class2embed = one_class2embed_kinetics700
    else:
        ValueError()

    embedding = []
    texts = []
    for class_name in class_name_inputs:
        tmp = one_class2embed(class_name, wv_model)
        embedding.append(tmp[0])
        texts.append(tmp[1])
    
    
    embedding = np.stack(embedding)
    
    return [normalize(embedding.squeeze()), texts]

def load_word2vec(curr_path):
    try:
        wv_model = Word2Vec.load(curr_path+'/GoogleNews', mmap='r')
    except:
        wv_model = Word2Vec.load_word2vec_format(
            curr_path+'/GoogleNews-vectors-negative300.bin', binary=True)
        wv_model.init_sims(replace=True)
        wv_model.save(curr_path+'/GoogleNews')

    # in case you are unable to download the word2vec automatically, you can download it from here 
    # https://drive.google.com/file/d/1-qGFAwGphw5bowrlm3aVVZmu4Y3VaGHg/view?usp=sharing
    # unzip and keep in the same path

    return wv_model

def one_class2embed_kinetics700(name, wv_model):
    return one_class2embed_kinetics400(name, wv_model)

def one_class2embed_kinetics400(name, wv_model):
    change = { # based on https://github.com/bbrattoli/ZeroShotVideoClassification
        'clean and jerk': ['weight', 'lift'],
        'dancing gangnam style': ['dance', 'korean'],
        'breading or breadcrumbing': ['bread', 'crumb'],
        'eating doughnuts': ['eat', 'bun'],
        'faceplanting': ['face', 'fall'],
        'hoverboarding': ['skateboard', 'electric'],
        'hurling (sport)': ['hurl', 'sport'],
        'jumpstyle dancing': ['jumping', 'dance'],
        'passing American football (in game)': ['pass', 'american', 'football', 'match'],
        'passing American football (not in game)': ['pass', 'american', 'football', 'park'],
        'petting animal (not cat)': ['pet', 'animal'],
        'punching person (boxing)': ['punch', 'person', 'boxing'],
        "massaging person's head": ['massage', "person", 'head'],
        'shooting goal (soccer)': ['shoot', 'goal', 'soccer'],
        'skiing (not slalom or crosscountry)': ['ski'],
        'throwing axe': ['throwing', 'ax'],
        'tying knot (not on a tie)': ['ty', 'knot'],
        'using remote controller (not gaming)': ['remote', 'control'],
        'backflip (human)': ['backflip', 'human'],
        'blowdrying hair': ['dry', 'hair'],
        'making paper aeroplanes': ['make', 'paper', 'airplane'],
        'mixing colours': ['mix', 'colors'],
        'photobombing': ['take', 'picture'],
        'playing rubiks cube': ['play', 'cube'],
        'pretending to be a statue': ['pretend', 'statue'],
        'throwing ball (not baseball or American football)': ['throw',  'ball'],
        'curling (sport)': ['curling', 'sport'],
    }
    
    if name in change:
        name_vec = change[name]
    else:
        name = name.lower()
        name_vec_origin = name.split(' ')
        remove = ['a', 'the', 'of', ' ', '', 'and', 'at', 'on', 'in', 'an', 'or',
                  'do', 'using', 'with']
        name_vec = [n for n in name_vec_origin if n not in remove]

        not_id = [i for i, n in enumerate(name_vec) if n == '(not']
        if len(not_id) > 0:
            name_vec = name_vec[:not_id[0]]
        name_vec = [name.replace('(', '').replace(')', '') for name in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec

def one_class2embed_ucf(name, wv_model):
    change = {
        'clean and jerk': ['weight', 'lift'],
        'skijet': ['Skyjet'],
        'handstand pushups': ['handstand', 'pushups'],
        'push ups': ['pushups'],
        'pull ups': ['pullups'],
        'walking with dog': ['walk', 'dog'],
        'throw discus': ['throw', 'disc'],
        'tai chi': ['taichi'],
        'cutting in kitchen': ['cut', 'kitchen'],
        'yo yo': ['yoyo'],
    }
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec

def one_class2embed_hmdb(name, wv_model):
    change = {'claping': ['clapping']}
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec

def one_class2embed_rareact(name, wv_model):
    name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def verbs2basicform(words):
    ret = []
    for w in words:
        analysis = wn.synsets(w)
        if any([a.pos() == 'v' for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'v')
        ret.append(w)
    return ret
