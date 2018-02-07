## Some imports
# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2

# gensim
from gensim.models import word2vec, KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.corpora import Dictionary
from gensim import corpora, models

# scipy
import scipy
from scipy.stats import kurtosis
from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, minkowski, braycurtis

# string distances
from fuzzywuzzy import fuzz
from pylev import levenshtein
from nltk.util import ngrams

# graph library
import networkx as nx
from collections import defaultdict

# intern scripts
from ner_features import *
from preprocess_questions import *
from constants import *
from stacking import *

# disable a few depreciation warnings
import warnings
warnings.filterwarnings('ignore')


# print(' .. Loading word2vec embeddings trained on corpus ..')
# model_w2v = KeyedVectors.load("300features_1minwords_5context")

def magic_features(train_data, test_data):
    ''' Compute common neigbours and frequency of the questions : highly predictive '''
    train = train_data.copy()
    test = test_data.copy()
    all_questions = pd.concat([train[['qid1', 'qid2']],
                               test[['qid1', 'qid2']]],
                              axis=0,
                              ignore_index=True)
    neighboors_dict = defaultdict(set)
    for i in range(all_questions.shape[0]):
        neighboors_dict[all_questions.qid1[i]].add(all_questions.qid2[i])
        neighboors_dict[all_questions.qid2[i]].add(all_questions.qid1[i])

    def q1_q2_neighboors(qid1, qid2):
        return len(neighboors_dict[qid1].intersection(neighboors_dict[qid2]))
    def frequency(qid1, qid2):
        f_1 = len(neighboors_dict[qid1])
        f_2 = len(neighboors_dict[qid2])
        return [min(f_1,f_2), max(f_1, f_2)] ## min-max conversion to ensure features are independent from the order of the questions
        
    return np.array([[q1_q2_neighboors(qid1, qid2)] + frequency(qid1, qid2) for qid1, qid2 in \
                zip(train['qid1'], train['qid2'])]), \
           np.array([[q1_q2_neighboors(qid1, qid2)] + frequency(qid1, qid2) for qid1, qid2 in \
                     zip(test['qid1'], test['qid2'])]) 


def graphical_features(train_data, test_data):
    ''' Compute some structural features on the graph obtained from the questions '''
    df = pd.concat([train_data[['qid1', 'qid2']], test_data[['qid1', 'qid2']]], axis = 0, ignore_index=True)
    g = nx.Graph()
    g.add_nodes_from(df.qid1)
    edges = list(df[["qid1", "qid2"]].to_records(index=False))
    g.add_edges_from(edges)
    g.remove_edges_from(g.selfloop_edges())
    print('Get kcore dict')
    kcore_dict = nx.core_number(g)
    print('Get centrality dict')
    centrality_dict = nx.degree_centrality(g)
    print('Get closeness dict')
    closeness_dict = nx.closeness_centrality(g)
    print('Get cliques dict')
    cliques_dict = nx.number_of_cliques(g)
    
    return np.array([(min(kcore_dict[qid1], kcore_dict[qid2]), max(kcore_dict[qid1], kcore_dict[qid2]),
     min(centrality_dict[qid1], centrality_dict[qid2]), max(centrality_dict[qid1], centrality_dict[qid2]),
     min(closeness_dict[qid1], closeness_dict[qid2]), max(closeness_dict[qid1], closeness_dict[qid2]),
     min(cliques_dict[qid1], cliques_dict[qid2]), max(cliques_dict[qid1], cliques_dict[qid2]))
     for qid1, qid2 in zip(train_data.qid1, train_data.qid2)]), \
           np.array([(min(kcore_dict[qid1], kcore_dict[qid2]), max(kcore_dict[qid1], kcore_dict[qid2]),
     min(centrality_dict[qid1], centrality_dict[qid2]), max(centrality_dict[qid1], centrality_dict[qid2]),
     min(closeness_dict[qid1], closeness_dict[qid2]), max(closeness_dict[qid1], closeness_dict[qid2]),
     min(cliques_dict[qid1], cliques_dict[qid2]), max(cliques_dict[qid1], cliques_dict[qid2]))
     for qid1, qid2 in zip(test_data.qid1, test_data.qid2)])


def question_flag_features(train_data, test_data):
    ''' Check the intent of the questions in the pairs, eg. "how"/"when" is likely to be non-duplicate '''
    train = train_data.copy()
    test = test_data.copy()
    
    def find_question_flag(w):
        #w = word_tokenize(s)
        for k in range(len(w)):
            if w[k] in question_flags: #.lower()
                return w[k] #.lower()
        
        return 'undetected' ## some questions do not have an explicit question flag

    train['q1_flag'] = train.q1p.apply(find_question_flag)
    train['q2_flag'] = train.q2p.apply(find_question_flag)
    test['q1_flag'] = test.q1p.apply(find_question_flag)
    test['q2_flag'] = test.q2p.apply(find_question_flag)
    train['pair_flags'] = train.apply(lambda s : str(sorted([s.q1_flag, s.q2_flag])), axis = 1)
    test['pair_flags'] = test.apply(lambda s : str(sorted([s.q1_flag, s.q2_flag])), axis = 1)
    flags_pairs_train = pd.get_dummies(train['pair_flags'].apply(pd.Series).stack()).sum(level=0)
    flags_pairs_test = pd.get_dummies(test['pair_flags'].apply(pd.Series).stack()).sum(level=0)
    for f in flags_pairs_test.columns:
        if f not in flags_pairs_train.columns:
            del flags_pairs_test[f]
    flags_pairs_test = flags_pairs_test.loc[:, flags_pairs_train.columns.tolist()]
    flags_pairs_test.fillna(value = int(0), inplace=True)

    fs = SelectKBest(chi2, 25) ## reduce the sparse features obtained with some chi2-test to get 25 most predictive features
    fp_train = fs.fit_transform(flags_pairs_train.values, train_data.is_duplicate.values)
    fp_test = fs.transform(flags_pairs_test.values)
    selected_pairs_flags = [e for ix,e in enumerate(flags_pairs_train.columns.tolist()) if fs.get_support()[ix]]

    return fp_train, fp_test, selected_pairs_flags

def last_character_features(train_data, test_data):
    ''' Check weither both questions finish the same way or not '''

    def last_character(q1,q2):
        q1_list = q1.replace(' ', '')
        q2_list = q2.replace(' ', '')
        try:
            if q1_list[-1] == q2_list[-1]:
                return 1
            else:
                return 0
        except:
            return 0

    return np.array([last_character(q1, q2) for q1, q2 in zip(train_data.q1, train_data.q2)]).reshape(-1,1),\
               np.array([last_character(q1, q2) for q1, q2 in zip(test_data.q1, test_data.q2)]).reshape(-1,1)


def w2v_metrics_features(train_data, test_data):
    ''' Compute some statistics about similarity between pairs of words in the w2vec embedding space '''

    def w2v_metrics(q1,q2):
        combinations = list(itertools.product([e for e in q1 if e not in stop_words], [e for e in q2 if e not in stop_words]))
        combinations = [list(combination) for combination in combinations]
        
        values = []
        for combination in combinations:
            try:
                values.append(model_w2v.similarity(combination[0], combination[1]))
            except KeyError:
                pass
  
        if(len(values) > 0):
            return [np.mean(values), np.median(values), np.std(values), kurtosis(values)]
        else:
            return [-1,-1,-1,-1]

    return np.array([w2v_metrics(q1, q2) for q1, q2 in tqdm(zip(train_data.q1p, train_data.q2p))]), \
           np.array([w2v_metrics(q1, q2) for q1, q2 in tqdm(zip(test_data.q1p, test_data.q2p))])

def glove_metrics_features(train_data, test_data):
    ''' Compute some statistics about similarity between pairs of words in the gloVe embedding space '''

    def w2v_metrics(q1,q2):
        combinations = list(itertools.product([e for e in q1 if e not in stop_words], [e for e in q2 if e not in stop_words]))
        combinations = [list(combination) for combination in combinations]
        
        values = []
        for combination in combinations:
            try:
                values.append(model_glove.similarity(combination[0], combination[1]))
            except KeyError:
                pass
  
        if(len(values) > 0):
            return [np.mean(values), np.median(values), np.std(values), kurtosis(values)]
        else:
            return [-1,-1,-1,-1]

    return np.array([w2v_metrics(q1, q2) for q1, q2 in tqdm(zip(train_data.q1p, train_data.q2p))]), \
           np.array([w2v_metrics(q1, q2) for q1, q2 in tqdm(zip(test_data.q1p, test_data.q2p))])


def fasttext_metrics_features(train_data, test_data):
    ''' Compute some statistics about similarity between pairs of words in the fastText embedding space '''

    def w2v_metrics(q1,q2):
        combinations = list(itertools.product([e for e in q1 if e not in stop_words], [e for e in q2 if e not in stop_words]))
        combinations = [list(combination) for combination in combinations]
        
        values = []
        for combination in combinations:
            try:
                values.append(model_fasttext.similarity(combination[0], combination[1]))
            except KeyError:
                pass
  
        if(len(values) > 0):
            return [np.mean(values), np.median(values), np.std(values), kurtosis(values)]
        else:
            return [-1,-1,-1,-1]

    return np.array([w2v_metrics(q1, q2) for q1, q2 in tqdm(zip(train_data.q1p, train_data.q2p))]), \
           np.array([w2v_metrics(q1, q2) for q1, q2 in tqdm(zip(test_data.q1p, test_data.q2p))])


def shared_words_features(train_data, test_data):
    ''' Compute shared n_grams between pairs of questions, up to 8 grams '''
    def common_ngrams(a, b, n):
        ngram_a = [" ".join(list(e)) for e in ngrams(a, n)]
        ngram_b = [" ".join(list(e)) for e in ngrams(b, n)]
        a_b = [e for e in ngram_a if e in ngram_b]
        _a = len(ngram_a)
        _b = len(ngram_b)
        _a_b = len(a_b)

        if _a==0 or _b==0:
            return 0, 0, 0
        
        return [2. * _a_b/(_a+_b), _a_b/min(_a, _b), _a_b/max(_a, _b)]


    def shared_words(q1,q2, process):
        if process:
            question1_words = q1.copy()
            question2_words = q2.copy()
        else:
            question1_words = list(set(q1.lower().split()))
            question2_words = list(set(q2.lower().split()))
        output = []
        for n in [1, 2, 3, 4, 5, 6, 7, 8]:
            output += common_ngrams(question1_words, question2_words, n)
        return output

    return np.concatenate([np.array([shared_words(q1, q2, process=False) for q1, q2 in zip(train_data.q1, train_data.q2)]),
                           np.array([shared_words(q1, q2, process=True) for q1, q2 in zip(train_data.q1p, train_data.q2p)])], axis = 1), \
           np.concatenate([np.array([shared_words(q1, q2, process=False) for q1, q2 in zip(test_data.q1, test_data.q2)]),
                           np.array([shared_words(q1, q2, process=True) for q1, q2 in zip(test_data.q1p, test_data.q2p)])], axis = 1)


def word_lengths_features(train_data, test_data):
    ''' Some basic features based upon length of questions, at character and word levels '''
    def word_lengths(q1,q2):

        #Feature: Character count of Questions
        len_char_q1 = len(q1.replace(' ', ''))
        len_char_q2 = len(q2.replace(' ', ''))
        len_char_diff = np.abs(len_char_q1 - len_char_q2)

        #Feature: Word count of Questions
        len_word_q1 = len(word_tokenize(q1))
        len_word_q2 = len(word_tokenize(q2))
        len_word_diff = np.abs(len_word_q1 - len_word_q2)

        #return len_char_q1, len_char_q2, len_char_diff, len_word_q1, len_word_q2, len_word_diff
        return min(len_char_q1, len_char_q2), max(len_char_q1, len_char_q2), len_char_diff, \
               min(len_word_q1, len_word_q2), max(len_word_q1, len_word_q2), len_word_diff

    return np.array([word_lengths(q1, q2) for q1, q2 in zip(train_data.q1, train_data.q2)]), \
           np.array([word_lengths(q1, q2) for q1, q2 in zip(test_data.q1, test_data.q2)])


def lev_distance_features(train_data, test_data):
    ''' Levenshtein distance between questions '''

    def lev_distance(q1, q2, process):
        if process:
            lev = float(levenshtein(' '.join(q1), ' '.join(q2)))
            return [lev / float(max(1, len(' '.join(q1)) + len(' '.join(q2)))),
                   lev / float(max(1, min(len(' '.join(q1)), len(' '.join(q2))))),
                   lev / float(max(1, max(len(' '.join(q1)), len(' '.join(q2)))))]

        else:
            lev = float(levenshtein(q1, q2))
            return [lev / float(max(1, len(q1) + len(q2))), 
                   lev / float(max(1, min(len(q1), len(q2)))),
                   lev / float(max(1, max(len(q1), len(q2))))]
    
    return np.concatenate([np.array([lev_distance(q1, q2, process=False) for q1,q2 in tqdm(zip(train_data.q1, train_data.q2))]),
                           np.array([lev_distance(q1, q2, process=True) for q1,q2 in tqdm(zip(train_data.q1p, train_data.q2p))])],
                          axis = 1), \
           np.concatenate([np.array([lev_distance(q1, q2, process=False) for q1,q2 in tqdm(zip(test_data.q1, test_data.q2))]),
                           np.array([lev_distance(q1, q2, process=True) for q1,q2 in tqdm(zip(test_data.q1p, test_data.q2p))])],
                          axis = 1)    

def vectorized_features(train_data, test_data, fitted_vectorizer, abs_diff = True, embedded = False):
    '''
         Transforms pairs of questions into the tfidf space and forms the absolute difference between them.
         Parameter embedded set to True will embed the difference vector into a 3-dimensional space
         with the predictions obtained from a stacking scheme (MultinomialNB, RandomForest, LogisticRegression)
         From my experience it is better not to embed them.
    '''
    train_q1 = list(train_data.q1p.apply(lambda s : ' '.join(s)).values)
    train_q2 = list(train_data.q2p.apply(lambda s : ' '.join(s)).values)
    test_q1 = list(test_data.q1p.apply(lambda s : ' '.join(s)).values)
    test_q2 = list(test_data.q2p.apply(lambda s : ' '.join(s)).values)

    if abs_diff:
        a = np.abs(fitted_vectorizer.transform(train_q1) - fitted_vectorizer.transform(train_q2))
        b = np.abs(fitted_vectorizer.transform(test_q1) - fitted_vectorizer.transform(test_q2))
        if not embedded: return a, b, embedded
        models = [MultinomialNB(alpha=1e-10), LogisticRegression(C = 35), RandomForestClassifier()]
        S_train, S_test = stacking(models, a, train_data.is_duplicate.values, b, n_folds = 10, verbose = 2)
        return S_train, S_test, embedded
    else:
        print('no abs diff')
        a = fitted_vectorizer.transform(train_q1) - fitted_vectorizer.transform(train_q2)
        b = fitted_vectorizer.transform(test_q1) - fitted_vectorizer.transform(test_q2)
        if not embedded: return a, b, embedded
        models = [GaussianNB(), LogisticRegression(C = 35), RandomForestClassifier()]
        S_train, S_test = stacking(models, a, train_data.is_duplicate.values, b, n_folds = 10, verbose = 2)
        return S_train, S_test, embedded

## An other version of the tfidf features bellow, transforms the questions and calculate a bunch of different metrics
## on them, cosine, euclidean, canberra, ...
## Long to compute and not so predictive, uncomment if you want to give it a try

# def vectorized_features(train_data, test_data, fitted_vectorizer):
#     train_q1 = list(train_data.q1p.apply(lambda s : ' '.join(s)).values)
#     train_q2 = list(train_data.q2p.apply(lambda s : ' '.join(s)).values)
#     test_q1 = list(test_data.q1p.apply(lambda s : ' '.join(s)).values)
#     test_q2 = list(test_data.q2p.apply(lambda s : ' '.join(s)).values)

#     v_train_1 = fitted_vectorizer.transform(train_q1)
#     v_train_2 = fitted_vectorizer.transform(train_q2)
#     v_test_1 = fitted_vectorizer.transform(test_q1)
#     v_test_2 = fitted_vectorizer.transform(test_q2)

#     del train_q1
#     del train_q2
#     del test_q1
#     del test_q2

#     print('Compute train cosine')
#     cosine_train = np.array([cosine(x.toarray(), y.toarray()) for x, y in zip(v_train_1, v_train_2)])
#     print('Compute train cityblock')
#     cityblock_train = np.array([cityblock(x.toarray(), y.toarray()) for x, y in zip(v_train_1, v_train_2)])
#     print('Compute train canberra')
#     canberra_train = np.array([canberra(x.toarray(), y.toarray()) for x, y in zip(v_train_1, v_train_2)])
#     print('Compute train euclidean')
#     euclidean_train = np.array([euclidean(x.toarray(), y.toarray()) for x, y in zip(v_train_1, v_train_2)])
#     print('Compute train braycurtis')
#     braycurtis_train = np.array([braycurtis(x.toarray(), y.toarray()) for x, y in zip(v_train_1, v_train_2)])
#     print('Compute train minkowski')
#     minkowski_train = np.array([minkowski(x.toarray(), y.toarray(), 3) for x, y in zip(v_train_1, v_train_2)])

#     print('Compute test cosine')
#     cosine_test = np.array([cosine(x.toarray(), y.toarray()) for x, y in zip(v_test_1, v_test_2)])
#     print('Compute test cityblock')
#     cityblock_test = np.array([cityblock(x.toarray(), y.toarray()) for x, y in zip(v_test_1, v_test_2)])
#     print('Compute test canberra')
#     canberra_test = np.array([canberra(x.toarray(), y.toarray()) for x, y in zip(v_test_1, v_test_2)])
#     print('Compute test euclidean')
#     euclidean_test = np.array([euclidean(x.toarray(), y.toarray()) for x, y in zip(v_test_1, v_test_2)])
#     print('Compute test braycurtis')
#     braycurtis_test = np.array([braycurtis(x.toarray(), y.toarray()) for x, y in zip(v_test_1, v_test_2)])
#     print('Compute test minkowski')
#     minkowski_test = np.array([minkowski(x.toarray(), y.toarray(), 3) for x, y in zip(v_test_1, v_test_2)])

#     return np.vstack([cosine_train, cityblock_train, canberra_train, euclidean_train, braycurtis_train, minkowski_train]).T, \
#            np.vstack([cosine_test, cityblock_test, canberra_test, euclidean_test, braycurtis_test, minkowski_test]).T

def ner_features(train_data, test_data):
    '''
         Checks weither questions are coherent according to the NER entities detected by Spacy.
         eg. if one question mentions 'India' and the other one 'Nigeria', questions are probably non-duplicate
         Implemented only for gpe (ie locations) NER for now
    '''
    return ner_gpe_features(train_data, test_data)

def words_mover_features(train_data, test_data):
    ''' Computes words mover distance between pair of questions and deduces a similarity measure '''
    def one_two_difference(s1, s2):
        if len(s1)!=len(s2): return 0
        else: return int(np.sum(np.array(s1) == np.array(s2)) in [len(s1)-1, len(s1) - 2])
    
    train = train_data.copy()
    test = test_data.copy()
    data = pd.concat([train[['q1p', 'q2p']], test[['q1p', 'q2p']]], axis = 0, ignore_index=True)
    sentences = list(data.q1p) + list(data.q2p)
    del data
    w2v = Word2Vec(size = 300, min_count = 1)
    w2v.build_vocab(sentences)

    train['wmd'] = train.progress_apply(lambda x : 1 - w2v.wv.wmdistance([e for e in x.q1p if e not in stop_words], [e for e in x.q2p if e not in stop_words]), axis = 1)
    test['wmd'] = test.progress_apply(lambda x : 1 - w2v.wv.wmdistance([e for e in x.q1p if e not in stop_words], [e for e in x.q2p if e not in stop_words]), axis = 1)

    t = 0.001
    ## Bins similarities in boxes of size t, and then correct the higly close pairs of question that has only one or two tokens difference :
    ## Their similarity is high but most of the time questions are not duplicate because a single detail varies from q1 to q2
    return train.apply(lambda s : .9815 if (.995 <= np.floor(s.wmd/t)*t <= .998) and ((one_two_difference(s.q1p, s.q2p)==1) \
                                                                                 or (same_pattern.search(s.q1) is not None)) \
                                                                                 else max(np.floor(s.wmd/t)*t, .976), axis = 1).values.reshape(-1, 1), \
           test.apply(lambda s : .9815 if (.995 <= np.floor(s.wmd/t)*t <= .998) and ((one_two_difference(s.q1p, s.q2p)==1) \
                                                                                   or (same_pattern.search(s.q1) is not None)) \
                                                                                else max(np.floor(s.wmd/t)*t, .976), axis = 1).values.reshape(-1, 1)

def fuzzy_features(train_data, test_data):
    ''' Compute fuzzywuzzy similarities between questions '''
    def fuzzy(q1, q2):
        return fuzz.token_set_ratio(q1, q2), fuzz.token_sort_ratio(q1, q2), fuzz.partial_token_sort_ratio(q1, q2), \
               fuzz.QRatio(q1, q2), fuzz.partial_ratio(q1, q2), fuzz.WRatio(q1, q2)

    return np.concatenate([np.array([fuzzy(q1, q2) for q1, q2 in zip(train_data.q1p.apply(lambda s : ' '.join(s)), train_data.q2p.apply(lambda s : ' '.join(s)))]),
                           np.array([fuzzy(q1, q2) for q1, q2 in zip(train_data.q1, train_data.q2)])], axis = 1), \
           np.concatenate([np.array([fuzzy(q1, q2) for q1, q2 in zip(test_data.q1p.apply(lambda s : ' '.join(s)), test_data.q2p.apply(lambda s : ' '.join(s)))]),
                                np.array([fuzzy(q1, q2) for q1, q2 in zip(test_data.q1, test_data.q2)])], axis = 1)



def spgk_features(train_data, test_data):
    ''' Computes some Shortest Path Graph Kernel distance on pairs of questions (graph of words) '''
    def create_graphs_of_words(docs, window_size):
        graphs = []
        sizes = []
        degs = []
        
        for doc in docs:
            G = nx.Graph()
            for i in range(len(doc)):
                if doc[i] not in G.nodes():
                    G.add_node(doc[i])
                for j in range(i+1, i+window_size):
                    if j < len(doc):
                        G.add_edge(doc[i], doc[j])

            graphs.append(G)
            sizes.append(G.number_of_nodes())
            degs.append(2.0*G.number_of_edges()/G.number_of_nodes())

        return graphs

    def spgk(sp_g1, sp_g2, norm1, norm2):
        if norm1 == 0 or norm2==0:
            return 0
        else:
            kernel_value = 0
            for node1 in sp_g1:
                if node1 in sp_g2:
                    kernel_value += 1
                    for node2 in sp_g1[node1]:
                        if node2 != node1 and node2 in sp_g2[node1]:
                            kernel_value += (1.0/sp_g1[node1][node2]) * (1.0/sp_g2[node1][node2])

            kernel_value /= (norm1 * norm2)
            return kernel_value

    def build_kernel_matrix(graphs, depth):
        sp = []
        norm = []

        for g in graphs:
            current_sp = dict(nx.all_pairs_dijkstra_path_length(g, cutoff=depth))
            sp.append(current_sp)
            
            sp_g = nx.Graph()
            for node in current_sp:
                for neighbor in current_sp[node]:
                    if node == neighbor:
                        sp_g.add_edge(node, node, weight=1.0)
                    else:
                        sp_g.add_edge(node, neighbor, weight=1.0/current_sp[node][neighbor])

            M = nx.to_numpy_matrix(sp_g)
            norm.append(np.linalg.norm(M,'fro'))

        K = np.zeros((len(graphs), len(graphs)))
        last_len = 0
        for i in range(len(graphs)):
            for j in range(i,len(graphs)):
                K[i,j] = spgk(sp[i], sp[j], norm[i], norm[j])
                K[j,i] = K[i,j]
        return K
    def SPGK_similarity(q1, q2, window_size=2, depth=1):
        return build_kernel_matrix(create_graphs_of_words([q1, q2], window_size), depth)[0,1]

    return np.array([SPGK_similarity(q1, q2) for q1, q2 in tqdm(zip(train_data.q1p, train_data.q2p))]).reshape(-1, 1), \
           np.array([SPGK_similarity(q1, q2) for q1, q2 in tqdm(zip(test_data.q1p, test_data.q2p))]).reshape(-1, 1)


class FeatureExtractor:
    ''' Class to perform the whole feature extraction process '''

    def __init__(self, vectorizers):
        self.vectorizers = vectorizers.copy()
        self.features_name = []
        pass

    def fit(self, train_data, test_data):
        all_questions = np.sum([list(train_data.q1p.apply(lambda s : ' '.join(s)).values),
            list(train_data.q2p.apply(lambda s : ' '.join(s)).values),
            list(test_data.q1p.apply(lambda s : ' '.join(s)).values),
            list(test_data.q2p.apply(lambda s : ' '.join(s)).values)])
        for ix, vectorizer in enumerate(self.vectorizers):
            print('.. Fitting vectorizer %d of %d ..' % (ix+1, len(self.vectorizers)))
            vectorizer.fit(all_questions)
        return self


    def transform(self, train_data, test_data):

        print(' .. Get last_character features ..')
        f_last_character_train, f_last_character_test = last_character_features(train_data, test_data)
        self.features_name.append('last_character_equal')

        print(' .. Get graphical features ..')
        f_graph_train, f_graph_test = graphical_features(train_data, test_data)
        self.features_name += ['min_core', 'max_core', 'min_centrality', 'max_centrality', 'min_closeness', 'max_closeness', 'min_cliques', 'max_cliques']

        print(' .. Loading w2vec pre trained embeddings ..')
        model_w2v = KeyedVectors.load_word2vec_format("/wordtovec/GoogleNews-vectors-negative300.bin", binary=True)
       
        print(' .. Compute w2v metrics ..')
        f_w2v_metrics_train, f_w2v_metrics_test = w2v_metrics_features(train_data, test_data)
        self.features_name.append('w2v_mean')
        self.features_name.append('w2v_median')
        self.features_name.append('w2v_std')
        self.features_name.append('w2v_kurtosis')
        
        del model_w2v

        print(' .. Loading gloVe pre trained embeddings ..')
        model_glove = KeyedVectors.load_word2vec_format('glove.txt') 

        print(' .. Compute glove metrics ..')
        f_glove_metrics_train, f_glove_metrics_test = glove_metrics_features(train_data, test_data)
        self.features_name.append('glove_mean')
        self.features_name.append('glove_median')
        self.features_name.append('glove_std')
        self.features_name.append('glove_kurtosis')

        del model_glove

        print(' .. Loading FastText pre trained embeddings ..')
        model_fasttext = KeyedVectors.load_word2vec_format("/fasttext/crawl-300d-2M.vec")

        print(' .. Compute fasttext metrics ..')
        f_fasttext_metrics_train, f_fasttext_metrics_test = fasttext_metrics_features(train_data, test_data)
        self.features_name.append('fasttext_mean')
        self.features_name.append('fasttext_median')
        self.features_name.append('fasttext_std')
        self.features_name.append('fasttext_kurtosis')

        del model_fasttext
        
        print(' .. Get shortest path graph kernel distances ..')
        f_spgk_train, f_spgk_test = spgk_features(train_data, test_data)
        self.features_name.append('spgk_distance')

        print(' .. Get question flags features ..')
        f_question_flag_train, f_question_flag_test, flags_list = question_flag_features(train_data, test_data)
        self.features_name += flags_list

        print(' .. Get shared words features ..')
        f_shared_words_train, f_shared_words_test = shared_words_features(train_data, test_data)
        self.features_name += [j for i in [['shared_'+str(i)+'gram_mean','shared_'+str(i)+'gram_min','shared_'+str(i)+'gram_max'] for i in [1, 2, 3, 4, 5, 6, 7, 8]] for j in i]
        self.features_name += [j for i in [['shared_'+str(i)+'gram_mean_processed','shared_'+str(i)+'gram_min_processed','shared_'+str(i)+'gram_max_processed'] for i in [1, 2, 3, 4, 5, 6, 7, 8]] for j in i]
        # self.features_name.append('shared_words')
        # self.features_name.append('shared_words_processed')

        print(' .. Get word lengths features ..')
        f_word_lengths_train, f_word_lengths_test = word_lengths_features(train_data, test_data)
        self.features_name.append('min_n_chars')
        self.features_name.append('max_n_chars')
        self.features_name.append('n_chars_diff')
        self.features_name.append('min_n_words')
        self.features_name.append('max_n_words')
        self.features_name.append('n_words_diff')

        print(' .. Compute Levenshtein distances ..')
        f_lev_distance_train, f_lev_distance_test = lev_distance_features(train_data, test_data)
        self.features_name.append('lev_distance_ratio')
        self.features_name.append('lev_distance_min')
        self.features_name.append('lev_distance_max')
        self.features_name.append('lev_distance_ratio_processed')
        self.features_name.append('lev_distance_min_processed')
        self.features_name.append('lev_distance_max_processed')

        print(' .. Get vectorizer features : 1 out of %d  ..' % len(self.vectorizers))
        f_vectorized_train, f_vectorized_test, embedded = vectorized_features(train_data, test_data, self.vectorizers[0], abs_diff=True)
        if embedded: self.features_name += ['vec1_pred_nb', 'vec1_pred_logit', 'vec1_pred_rf']
        else: self.features_name += ['vec1_' + e for e in self.vectorizers[0].get_feature_names()]
        # # f_vectorized_train, f_vectorized_test = vectorized_features(train_data, test_data, self.vectorizers[0])
        # # self.features_name += ['vec_1' + e for e in ['cosine', 'cityblock', 'canberra', 'euclidean', 'braycurtis', 'minkowski']]

        for ix, vectorizer in enumerate(self.vectorizers[1:]):
            print(' .. Get vectorizer features : %d out of %d  ..' % ((ix+2), len(self.vectorizers)))
            v_train, v_test, embedded = vectorized_features(train_data, test_data, vectorizer, abs_diff=True)
            f_vectorized_train = scipy.sparse.hstack(blocks = [f_vectorized_train, v_train])
            f_vectorized_test = scipy.sparse.hstack(blocks = [f_vectorized_test, v_test])
            if embedded: self.features_name += ['vec'+str(ix+2)+'_'+'pred_nb', 'vec'+str(ix+2)+'_'+'pred_logit', 'vec'+str(ix+2)+'_'+'pred_rf']
            else: self.features_name += ['vec' + str(ix+2) +'_' + e for e in vectorizer.get_feature_names()]
        #     # v_train, v_test = vectorized_features(train_data, test_data, vectorizer)
        #     # f_vectorized_train = np.hstack([f_vectorized_train, v_train])
        #     # f_vectorized_test = np.hstack([f_vectorized_test, v_test])
        #     # self.features_name += ['vec' + str(ix+2) +'_' + e for e in ['cosine', 'cityblock', 'canberra', 'euclidean', 'braycurtis', 'minkowski']]


        print(' .. Get NER features ..')
        f_ner_train, f_ner_test = ner_features(train_data, test_data)
        self.features_name.append('gpe_coherence')
        self.features_name.append('gpe_incoherence')

        print(" .. Compute word's mover affinity ..")
        f_wmd_train, f_wmd_test = words_mover_features(train_data, test_data)
        self.features_name.append('words_mover_affinity')

        print(' .. Get magic features ..')
        f_magic_train, f_magic_test = magic_features(train_data, test_data)
        self.features_name.append('common_neighboors')
        self.features_name.append('min_freq')
        self.features_name.append('max_freq')

        print(' .. Get fuzzy features ..')
        f_fuzzy_train, f_fuzzy_test = fuzzy_features(train_data, test_data)
        self.features_name.append('fuzzy_set_ratio_processed')
        self.features_name.append('fuzzy_sort_ratio_processed')
        self.features_name.append('fuzzy_partial_sort_ratio_processed')
        self.features_name.append('fuzzy_qratio_processed')
        self.features_name.append('fuzzy_partial_ratio_processed')
        self.features_name.append('fuzzy_wratio_processed')
        self.features_name.append('fuzzy_set_ratio')
        self.features_name.append('fuzzy_sort_ratio')
        self.features_name.append('fuzzy_partial_sort_ratio')
        self.features_name.append('fuzzy_qratio')
        self.features_name.append('fuzzy_partial_ratio')
        self.features_name.append('fuzzy_wratio')

        try:
            return scipy.sparse.hstack(blocks = [f_last_character_train,
                                                   f_graph_train,
                                                 f_w2v_metrics_train,
                                                 f_glove_metrics_train,
                                                 f_fasttext_metrics_train,
                                                 f_spgk_train,
                                                 f_question_flag_train,
                                                 f_shared_words_train,
                                                 f_word_lengths_train,
                                                 f_lev_distance_train,
                                                 f_vectorized_train,
                                                 f_ner_train,
                                                 f_wmd_train,
                                                 f_magic_train,
                                                 f_fuzzy_train]), \
                    scipy.sparse.hstack(blocks = [f_last_character_test,
                                                    f_graph_test,
                                                  f_w2v_metrics_test,
                                                  f_glove_metrics_test,
                                                  f_fasttext_metrics_test,
                                                  f_spgk_test,
                                                  f_question_flag_test,
                                                  f_shared_words_test,
                                                  f_word_lengths_test,
                                                  f_lev_distance_test,
                                                  f_vectorized_test,
                                                  f_ner_test,
                                                  f_wmd_test,
                                                  f_magic_test,
                                                  f_fuzzy_test])

        except:
            return np.concatenate([f_last_character_train,
                                 f_graph_train,
                               f_w2v_metrics_train,
                               f_glove_metrics_train,
                               f_fasttext_metrics_train,
                               f_spgk_train,
                               f_question_flag_train,
                               f_shared_words_train,
                               f_word_lengths_train,
                               f_lev_distance_train,
                               f_vectorized_train,
                               f_ner_train,
                               f_wmd_train,
                               f_magic_train,
                               f_fuzzy_train], axis = 1), \
                np.concatenate([f_last_character_test,
                                  f_graph_test,
                                f_w2v_metrics_test,
                                f_glove_metrics_test,
                                f_fasttext_metrics_test,
                                f_spgk_test,
                                f_question_flag_test,
                                f_shared_words_test,
                                f_word_lengths_test,
                                f_lev_distance_test,
                                f_vectorized_test,
                                f_ner_test,
                                f_wmd_test,
                                f_magic_test,
                                f_fuzzy_test], axis = 1)


    def get_feature_names(self):
        return self.features_name
