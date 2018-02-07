from nltk.corpus import stopwords
import string
import re


#Set parameters for word2vec model
num_features = 300 #Word vector dimensionality
min_word_count = 1 #Minimum word count 
num_workers = 4 #Number of threads to run in parallel
context = 5 #Context window size
downsampling = 1e-3 #Downsample setting for frequent words

model_name = "%dfeatures_%dminwords_%dcontext" % (num_features, min_word_count, context)


#Set parameters for preprocessing
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
punctuation.update(["``", "`", "..."])


#Set parameters for features construction
# question_flags = set(['what', 'how', 'why', 'which', 'is', 'can', 'who', "what",
#        'where', 'do', 'does', 'will', 'are', 'should', 'when', 'if',
#        'has', 'did', 'have', 'would', 'could', 'had'])

question_flags = set(['what', 'how', 'why', 'which', 'be', 'can', 'who', "what",
       'where', 'do', 'will', 'should', 'when', 'if',
       'have', 'would', 'could'])

saltwater = re.compile(r'salt[\s]*water')
safety_precaution = re.compile(r'safety precaution')
quickbook = re.compile(r'quickbook')
same_pattern = re.compile(r'employees should know|regulations when visiting|greenlit')

remove_gpe = set(['advaxis', 'unroll.me', 'snorlax', 'myriad', 'verastem', 'runescape', 'covanta', 'the liberty dollar', 'youtube video', 'bruker', 'escorts', 'bothell', 
    'windstream', 'olacab', 'whatsapp qr', 'f1', 'android os', 'sociopath', 'matlab?', 'thoreau', 'download', 'gardens', 'loc', 'cyberonics', 'hindi and urdu', 'cinton', 'dimension', 
    'nagaland', 'panipat', 'fedex', 'browsing', 'mussoorie', 'medgenics', 'bikes', 'drano', 'qm', 'maxlinear', 'mercedes benz', 'nougat', '9th', 'the soviet space program', 'morehouse', 
    'civil', 'pokémon', 'b.tech', 'youtube', 'bulldog', 'siri', 'evo vr', 'the mongol empire', '₹500', 'pl', 'qa', 'google', 'hillary', 'uae', 'firozabad', 'tweet', 'duterte’s', 'json', 
    'the battle of dewar', 'castles', 'masago', 'code academy', 'carbonite', 'endocyte', 'hyderebad', 'usb 3.0', 'yelp', 'incineroar', 'antminer', 'nyquil', 'sebi', 'rolex', 'interstellar',
    'headshot', 'b.a.', 'talky', 'axiall', 'inbox', 'ms', 'lt', 'demat', 'therapeuticsmd', 'assam', 'usd', 'scorpio', 'town', 'philosophy', 'quadrotor', 'moto g', 'ht', 'toefl', 'tesla x',
    'pomodoro', 'w3', 'globant', 'maui', 'evernote', 'pocket', 'ola', 'lynda', 'duolingo', 'messi', 'gbp', 'sc', 'local', 'arbok', 'android os', 'deducieye', 'cepheid', 'ebix', 'unclos', 
    'amdocs', 'epival', 'snapchats', 'netflix', 'earthquakes', 'timespro', 'etherium', 'spotify', 'belkin dsl', 'xanax', 'mars', 'if3', 'chattooga county', 'centerpoint energy', 'vlogging', 
    'ebay india', 'wattpad', 'lafayette', 'tazewell county', 'codecademy', 'sensex', 'tristan', 'extrovert', 'keepsafe', 'leave', 'dragon', 'chemtura', 'mcu phase', 'g.s.', 'toefl', 'kochi', 
    'dj', 'asansol', 'medicine', 'diglett', 'mechatronics', 'waitlist tatkal', 'asoiaf/', 'pm', 'youtube video', 'hallelujah', 'pearl', 'waitlist', 'multithreading', 'dancing', 'makemytrip', 
    'android l', 'rio olympics', 'corvel', 'saskatchewan', 'livestream', 'intermediate', 'mountain', '₹20000', 'beagle', 'mineral county', 'palantir', 'leicester city fc', 'craigslist', 
    'codehs', 'pinterest', 'depomed', 'dropbox', 'starboy', 'ola cab', 'trustmark', 'big lots', 'jira', 'ebay-india', 'denali county', 'tobiko', 'pn', 'tripadvisor', 'pokémon ranger', 
    'techcrunch', 'west bank', 'gokul', 'ecommerce', 'wrangell county', 'express', 'brain', 'rockmelt', 'linux', 'korra', 'paypall', 'undead', 'thc', 'periscope', 'wot', 'intralinks', 'hybris', 
    'the liberty dollar', 'macau', 'pomona', 'uber', 'odisha', 'homosexuality', 'eragon', 'marijuana', 'quora', 'hadoop', 'shoretel', 'wikipedia', '₹2000', 'bhel', 'guidestar', 'zeref', 
    'awesong', 'assange','chihuahua','pokemon ranger', 'electrical', 'aliens', 'clikbank', 'sociopaths', 'ee', 'konkani', 'aikido', 'josaa', 'nidoran', 'amazon prime', 'tamilnaadu', 
    'roanoke', 'humanities', 'b.e.', 'littelfuse', 'adtran', 'maroubra', 'hitchhiker', 'gmat', 'broward college', 'the city of heavenly fire', 'java', 'actuant', 'end', 'brazzers.com', 
    'bozeman', 'spider-man', 'xperia t2 ultra', 'cc', 'illnesses', 'miktex', 'ranchi', 'sbh', 'socialtrade.biz', 'instagram', 'eskimos', 'shakesperean', 'iim?(see', 'unilife', 
    'prison break', 'hortonworks', 'mophie iphone', 'sasuke', 'u.p.', 'emmys', 'angularjs', 'keiretsu', 'topcoder', 'cubone', 'chiloé', 'node.js', 'lenovo 4', 'indore', 'download', 
    'universitis', 'idaho state', 'thai', 'netsuite', 'laughlin', 'vs', 'taipei', 'kissanime', 'setturu', 'illuminati', 'upvotes', 'claremont colleges', 'cannabis', 'republic city', 
    'bharat ratna', 'vr', 'octonauts', 'imperva', 'genpact', 'tizen 3', 'irctc', 't.v.', 'pinterest', 'gmail', 'attack', 'msc industrial', 'miktek', 'wl', 'rajya sabha', 'bern', 'b.tech', 
    'shippûden', 'accenture', 'hindi', 'wechat', 'prestashop', 'rs.50000', 'parenting', 'telfair county', 'o+', 'talaq', 'eminem', 'github', 'oclaro', 'macbook pro', 'furmanite', 
    'mount etna', 'google+', 'swaraj', 'blockchain', 'android', 'ct', 'theri', 'madurai', 'xenoport', 'juno therapeutics', 'williamson county', 'gt', 'whiplash', 'tredegar', 'kitkat4.4.2', 
    'opencart', 'pokemon', 'haringey', 'inbox', 'lunala', 'logitech', 'icloud', 'itron', 'kibana', 'cph4', 'me', 'mount vesuvius', 'shimla', 'xtrade', 'shirdi', 'gajendra', 'backlink', 
    "cat'15", 'innings', 'putonghua', 'googling', 'deadpool', 'haradrim', 'po' 'id', 'bitch', 'freshpet', 'bhubaneswar?', 'pusheen facebook', 'balchem', 'parkinson', 'prakrit', 'ka', 
    'mankind', 'facebook', 'tambrahm', 'nayantara', 'psychopaths', 'gr', 'yahoo', 'sbi ppf', 'fuzhou', 'celgene', 'windows os', 'maths', 'twitter', 'delek', 'guerlain', 'skullcandy', 
    'capgemini', 'volte' 'punjab province of', 'startup', 'teletech', 'backtesting', 'peninsula county', "new brunswick's", 'macbook', 'tamilan', 'zogenix', 'civeo', 'north eastern', 
    'brown county', '6s', 'netapp', 'laravel', 'wabash national', 'marshmallow', 'manaphy', 'pf', 'dc', 'metrocard', 'templates', 'freemasonry', 'neograft fue', 'javascript codehs', 
    'jinnah', 'upwork', 'isro', 'iroquois tribe', 'fibonacci', 'borax', 'photoshop', 'snapchat', 'kalamazoo', 'partnerre', 'vedas', 'trovagene', 'chuka beach', 'nisource', 'quizcraft', 
    'redmi', 'sinus', 'agricultural', 'samoa'])