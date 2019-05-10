## altegrad_challenge

My solution for the altegrad challenge (January 2017)

I got 2nd on the private LB (Logloss : 0.13064).

## short description

See report.pdf for more details, but in brief :

It was copied from the Quora Challenge on Kaggle : the aim is to predict whether two short questions have the same meaning or not.

# preprocessing

NLTK, inflect, and some handcrafted instructions are used to normalize the questions as much as posible (strip accents and weird encodings, replace expressions, correct typos, lemmatize verbs to their infinitive form, set plurial nouns to singular, ...)

# feature extraction

My solution mixes NLP features :
- shared words features
- levenshtein distances
- words mover distances
- fuzzy similarities
- cosine similarities between embeddings from word2vec, glove and fastText
- stacking with the TFIDF / TWIDF vectorized representations of questions
- discordance of question flags
- Spacy NER tags
- ... 

along with non NLP features : 
- Shortest Path Graph Kernal on GraphOfWords representation of questions
- structural features on the whole graph formed by the pairs of questions such as K-cores, common neighboors, frequencies, closeness, centrality, number of cliques, ...)

obtained upon both the processed and non-processed versions of the questions

# models

The features are then fed to lightGBM tuned by Bayesian Optimization.

The embeddings are also fed to a combinason of LSTM and MLP layers.

Finally the two models are assembled and some post processing is done to boost the scores (log-loss)
