import re
import pandas as pd
import spacy
import en_core_web_sm
from constants import *

nlp = en_core_web_sm.load()

def ner_gpe_coherence(raw):
    
    if (saltwater.search(raw.q1.lower()) is not None) and (saltwater.search(raw.q2.lower()) is not None):
        return 'GPE_COHERENT'
    if (safety_precaution.search(raw.q1.lower()) is not None) and (safety_precaution.search(raw.q2.lower()) is not None):
        return 'GPE_COHERENT'
    if (quickbook.search(raw.q1.lower()) is not None) and (quickbook.search(raw.q2.lower()) is not None):
        return 'GPE_COHERENT'
    
    gpe_1 = [e.text.lower() for e in nlp(raw.q1.replace('the Pakistanis', 'Pakistan').replace('Uri attacks',\
                'attacks in Uri').replace('Uri attack', 'attacks in Uri').replace('uri attack', 'attacks in Uri').replace('uri attack', 'attacks in Uri').replace('india', \
                'India').replace('indians', 'India').replace('Indians', 'India').replace('indian', 'India').replace('varanasi', 'Varanasi').replace('Indian', \
                'India').replace('dellhi', 'delhi').replace('china', 'China')).ents if e.label_ == 'GPE']
    gpe_2 = [e.text.lower() for e in nlp(raw.q2.replace('the Pakistanis', 'Pakistan').replace('Uri attacks',\
                'attacks in Uri').replace('Uri attack', 'attacks in Uri').replace('uri attack', 'attacks in Uri').replace('uri attack', 'attacks in Uri').replace('india', \
                'India').replace('indians', 'India').replace('Indians', 'India').replace('indian', 'India').replace('varanasi', 'Varanasi').replace('Indian', \
                'India').replace('dellhi', 'delhi').replace('china', 'China')).ents if e.label_ == 'GPE']
    
    gpe_1 = set([e.replace('the united states of america', 'us').replace('the united states', \
            'us').replace('united states', 'us').replace('america', 'us').replace('u.s.a.', \
            'usa').replace('u.s.a', 'us').replace('usa', 'us').replace('u.s.', 'us').replace('u.s', \
            'us').replace('baluchistan', 'balochistan') for e in gpe_1 if e not in remove_gpe])
    gpe_2 = set([e.replace('the united states of america', 'us').replace('the united states', \
            'us').replace('united states', 'us').replace('america', 'us').replace('u.s.a.', \
            'usa').replace('u.s.a', 'us').replace('usa', 'us').replace('u.s.', 'us').replace('u.s', \
            'us').replace('baluchistan', 'balochistan') for e in gpe_2 if e not in remove_gpe])
    if ((len(gpe_1)==0) and (len(gpe_2)==0)) or ((list(gpe_1) == ['india']) and (list(gpe_2) == [])) \
            or ((list(gpe_2) == ['india']) and (list(gpe_1) == [])):
        return 'GPE_IRRELEVANT'
    if gpe_1.intersection(gpe_2) == gpe_1.union(gpe_2):
        return 'GPE_COHERENT'
    else:
        return 'GPE_INCOHERENT'


def ner_gpe_features(train_data, test_data):

    return pd.get_dummies(train_data.apply(ner_gpe_coherence, axis = 1, raw = True)).loc[:, ['GPE_COHERENT', 'GPE_INCOHERENT']].values, \
            pd.get_dummies(test_data.apply(ner_gpe_coherence, axis = 1, raw = True)).loc[:, ['GPE_COHERENT', 'GPE_INCOHERENT']].values