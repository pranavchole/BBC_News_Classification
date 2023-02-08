#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[62]:


print("Training set:")
train_df = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Train.csv")
test_df = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Test.csv")
sample_solution_df = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Sample Solution.csv")
train_df.head()


# In[63]:


print("Test set:")
test_df


# In[64]:


print("Sample solution set:")
sample_solution_df


# In[65]:


print('Inspect data types and nmissing values per column:')
train_df.info()


# In[66]:


print('Checking number of unique articles in training set:')
train_df.nunique()


# In[223]:


#-------------------- Helper functions for histogram count annotation -----------------------------
def add_histogram_values(ax): [ax.bar_label(remove_0_tags_for_histograms(b)) for b in ax.containers]
def remove_0_tags_for_histograms(ax_container):
    ind = np.where(ax_container.datavalues>0)[0]    
    ax_container.datavalues = ax_container.datavalues[ind]
    ax_container.patches = [ax_container.patches[i] for i in ind]
    return ax_container
#---------------------------------------------------------------------------------------------------
print('Number of articles per topic:')
fig, ax = plt.subplots(figsize=(8, 5))
ax2 = sns.histplot(
    data = train_df,
    x = 'Category',
    hue = 'Category',
    palette = 'colorblind',
    legend = False,
    ).set(
        title = 'Category Counts');
add_histogram_values(ax)
C = pd.DataFrame(train_df['Category'].value_counts())
C


# In[75]:


# Add column with multiplicity (note: leave it as a string for a prettier plot)
train_df['text multiplicity'] = train_df.groupby('Text')['Text'].transform('count').astype(str) 

print('What categories contain most of the repeated Text?')
fig, ax = plt.subplots(ncols=2,figsize=(18, 5))
# Histogram showing repeated text multiplicity in training dataframe
ax2 = sns.histplot(
    ax = ax[0],
    data = train_df,
    x='text multiplicity',
    palette = 'colorblind',
    multiple = 'dodge',
    legend = True,
    ).set(
        title = 'Multiplicity of text in articles');
add_histogram_values(ax[0])
# Same histogram, by Category
ax2 = sns.histplot(
    ax = ax[1],
    data = train_df,
    x='text multiplicity',
    hue = 'Category',
    palette = 'colorblind',
    multiple = 'dodge',
    legend = True,
    ).set(
        title = 'Multiplicity of text in articles by category');
add_histogram_values(ax[1])


# In[76]:


# Check how many categories are in each repeated text group
print('The maximum number of different categories assigned to repeated texts is: {}'.format(\
train_df[train_df['text multiplicity']!='1'].groupby('Text')['Category'].nunique().max()))


# In[77]:


train_df = train_df.drop_duplicates(subset=['Text'])
print('Number of articles per topic (without duplicates):')
fig, ax = plt.subplots(figsize=(8, 5))
ax2 = sns.histplot(
    data = train_df,
    x = 'Category',
    hue = 'Category',
    palette = 'colorblind',
    legend = False,
    ).set(
        title = 'Category Counts (without duplicates)');
add_histogram_values(ax)
pd.DataFrame(train_df['Category'].value_counts())


# In[226]:


print('*'*40)
print('Sample text:')
print('*'*40)
print(train_df['Text'][0])

print('\n'+'*'*40)
print('Frequency of punctuation marks in sample text:')
print('*'*40)
[print("  {} is {} times".format(c,train_df['Text'][0].count(c))) for c in ['.',',',';','?',':','!','"',"'",")","("]];

from collections import Counter
print('\n'+'*'*40)
print('Most common words in sample text:')
print('*'*40)
for word, count in Counter(train_df['Text'][0].split()).most_common(5):
    print("  '{}' is {} times".format(word, count))


# In[82]:


import re
def clean_text(df,keep_dots=False):
    """ cleans the column 'Text' in a DataFrame"""
    if keep_dots:
        clean_method = lambda x: clean_string(x,True)
    else:
        clean_method = clean_string
                
    try: # DataFrame
        return df['Text'].apply(clean_method)
    except KeyError: # Series
        return df.apply(clean_method)
    
def clean_string(s,keep_dots=False):
    """Cleans a string"""
    s = re.sub("\d+", " number ", s) # change numbers to word " number "
    if keep_dots:
        s = s.translate(s.maketrans("£,;:-", "$    ","()+-*!?%")) # replace £ with $ and remove punctuation       
    else:
        s = s.translate(s.maketrans("£.,;:-", "$     ","()+-*!?%")) # replace £ with $ and remove punctuation        
    s = s.replace("$ number", "money")
    s = s.replace("number bn", "money")
    s = s.replace("money bn", "money")
    s = s.replace("money   money", "money")
    s = s.replace("money money", "money")
    s = s.replace("number   number", "number")
    s = s.replace("number number", "number")
    return s


# In[83]:


# Clean training set
pd.options.mode.chained_assignment = None
train_df['Text']=clean_text(train_df)
pd.options.mode.chained_assignment = "warn"

print('*** 3 Sample texts after cleaning: ***')
[print(train_df['Text'][i]+'\n') for i in range(3)];


# In[84]:


pd.options.mode.chained_assignment = None
train_df['word count'] = train_df['Text'].apply(lambda row: len(row.split()))
train_df['char count'] = train_df['Text'].apply(lambda row: len(row))
pd.options.mode.chained_assignment = "warn"
print('Training set including number of word and number of characters per entry:')
train_df


# In[85]:


print('Statistics about characters and word count in training set')

fig, ax = plt.subplots(figsize=(15, 5))
sns.histplot(
    data = train_df, 
    x = 'word count',
    palette = 'colorblind',
    kde=True,
    ).set(
        title = 'Number of Words per Article');
train_df.describe()


# In[86]:


print('Word count statistics by category:')
G = train_df[['Category','word count']].groupby('Category')
G.describe()


# In[87]:


fig, ax = plt.subplots(figsize=(15, 5))
sns.histplot(
    data = train_df, 
    x = 'word count',
    palette = 'colorblind',
    kde=True,
    hue = 'Category'
    ).set(
        title = 'Number of words per article by category');

# words per category
fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(
    data = train_df, 
    x = 'word count', 
    y = 'Category',
    palette = 'colorblind'
    ).set(
        title = 'Number of words per article by category');


# In[89]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer() 
WC=cv.fit_transform(train_df['Text'])
print('There are {} documents in the training set.'.format(len(train_df)))
print('There are {} different words in the training set.'.format(WC.shape[1]))
print('The sparse Matrix WC has size {} and its sparsity is {:.2f}%. '.format(WC.shape,100*(1-WC.count_nonzero()/np.prod(WC.shape))))
print('\nCorresponding DataFrame to WC:')
counts_df = pd.DataFrame(WC.A,columns=cv.get_feature_names_out ())
counts_df


# In[90]:


print('The most popular words amog all documents are are:')
print('\nword  count')
print(counts_df.sum().sort_values().tail())


# In[91]:


# Include category column into word count
counts_df['_category_']=train_df['Category']

print('Word count by category:')
G = counts_df.groupby('_category_').sum()
G


# In[92]:


is_unique = G.astype(bool).sum(axis=0)==1

total_word_count = G.sum(axis=0)

a = total_word_count[is_unique]
b = G.idxmax(axis=0)[is_unique]
c = cv.get_feature_names_out ()[is_unique]

unique_words =  pd.DataFrame(data=zip(a,b),index=c,columns=['unique word count','Category'])
print('Unique words per category:')
unique_words.groupby('Category').describe()


# In[96]:


print('Top 10 Unique words and their count per category:')
C = unique_words.groupby('Category')
U = C.max()
U['top unique word'] = C.idxmax().iloc[:,0]
U
for category,g, in C:
    print('\nCategory: ',category,'\n',g['unique word count'].nlargest(10).to_markdown())  


# In[99]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer() 
WC2 = tfidf_transformer.fit_transform(WC)

tfidf_df = pd.DataFrame(WC2.A,columns = cv.get_feature_names_out ())
print('WC2 matrix representation:')
tfidf_df


# In[100]:


print('Raking of word scores in WC2:')
print('\n->column sum:')
print(tfidf_df.sum().sort_values())


# In[101]:


print("Weights given to words by idf:")
pd.DataFrame(tfidf_transformer.idf_,
             cv.get_feature_names_out (),
             columns=["idf_weights"]).sort_values(by="idf_weights")


# In[102]:


# Divide the words into 3 categories (unique,intermideate,universal) and merge them into a dataframe with column type:

print("Subset of the words that are in all categories (universal words):")
is_in_all_cats = G.astype(bool).sum(axis=0)==5
universal_words = pd.DataFrame(WC.getnnz(axis=0)[is_in_all_cats], # nnz = number of nonzero entries per column (count of articles with word)
                               index=G.columns[is_in_all_cats]._data,
                               columns=['#of Articles with this word'])


fig, axs = plt.subplots(ncols=1,figsize=(10, 5))
sns.histplot(
    ax=axs,
    data = universal_words, 
    legend = False,
    palette = 'colorblind',
    kde = True,
    ).set(
        title = 'Presence of universal words in articles',
        xlabel='Number of articles with this word');
universal_words.describe()


# In[103]:


print('These are the 50 most common universal words:')
print(universal_words.iloc[:,0].nlargest(50))
print("\nRemark: The word money is found in {} articles".format(universal_words.loc['money'][0]))


# In[104]:


# Compute which words are unique, intermediate or universal

universal_words['type']='universal'
is_interm_freq_word = ~is_in_all_cats & ~is_unique
intermediate_words =  pd.DataFrame(WC.getnnz(axis=0)[is_interm_freq_word], # nnz = number of nonzero entries per column (count of articles with word)
                               index=G.columns[is_interm_freq_word]._data,
                               columns=['#of Articles with this word'])
intermediate_words['type']='intermediate'

unique_words =  pd.DataFrame(WC.getnnz(axis=0)[is_unique], # nnz = number of nonzero entries per column (count of articles with word)
                               index=G.columns[is_unique]._data,
                               columns=['#of Articles with this word'])
unique_words['type']='unique' #defined in a previous cell.

word_types = pd.concat([unique_words,intermediate_words,universal_words],axis=0)


fig, axs = plt.subplots(ncols=1,figsize=(10, 5))
k=0
hue_order=['unique','intermediate','universal']
s=sns.ecdfplot(data = word_types,
             x= '#of Articles with this word',
             hue = "type",
             hue_order=hue_order,
             legend = False,                
             log_scale=(True,False),
#              label = hue_order
            ).set(
        title = 'Cumulative distribution of words by type',
        xlabel='Number of articles with this word');

axs.grid(visible=True, which='major', color='black', linewidth=0.2)
axs.grid(visible=True, which='minor', color='black', linewidth=0.075)
axs.legend(labels=hue_order[::-1])
word_types = word_types.groupby('type')
print('\nPresence of words in articles by word type:\n')
word_types.describe()


# In[105]:


def get_word_type_and_count(word:str, word_types:pd.DataFrame):
    """ Returns a pandas series with word type and #of Articles with this word"""
    for word_type, group_df in word_types:
        try:
            return group_df.loc[word,:]
        except KeyError:
            pass
def print_word_type_and_count(word:str, word_types:pd.DataFrame):
    word_info = get_word_type_and_count(word, word_types)
    print("The word '{}' is of {} type, found in {} articles.".format(
        word, word_info['type'], word_info['#of Articles with this word']))     

    print('Boxplots of word count per category for chosen words:')
fig, axs = plt.subplots(ncols=2,figsize=(16, 5));
words = ('money','number')
for i,word in enumerate(words):
    word_type = get_word_type_and_count(word,word_types)['type'];
    print_word_type_and_count(word,word_types)
    
    sns.boxplot(        
        ax = axs[i],
        data = counts_df, 
        x = word, 
        y = '_category_',
        palette = 'colorblind',    
        ).set(
            title = "Count of word '{}' per article (tpye='{}')".format(word,word_type),
            xlabel = 'Frequency of word within articles',
            ylabel = 'Category');


# In[108]:


from sklearn.model_selection import train_test_split
train_df = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Train.csv").drop_duplicates(subset=['Text'])
test_df = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Test.csv")
train_df, crossval_df = train_test_split(train_df,stratify=train_df[['Category']],test_size=0.2,random_state = 310)
print('Training set:')
train_df


# In[109]:


print('Cross validation set:')
crossval_df


# In[113]:


from sklearn.feature_extraction.text import TfidfVectorizer
class MyData(object):
    """
    Represent text data as a sparse matrix with TfidfVectorizer
    Note: This class does not follow sklearn syntax. 
    """
    def __init__(self,df,TfidfVec=None,clean_text=True,**TtdifVec_kwargs):
        """
        Inputs:
        -df: a DataFrame including column "Text"
        -TfidfVec: previously fitted TfidfVectorizer object (pass None for a new fit)        
        -clean_text: if True, 'Text' column will be cleaned.
        -TtdifVec_kwargs: parameters to pass when the TfidfVectorizer is created
        Fields:
        -MyData.words: list of words used by TfidfVec
        -MyData.tv: fitted TfidfVectorizer object
        -MyData.WC2: Sparse matrix representation of the text
        """
        self.df = df.copy()
        if clean_text:
            self.clean_text()
        self.get_WC2(TfidfVec,**TtdifVec_kwargs)
    def clean_text(self):
        self.df['Text'] = clean_text(self.df)
    def get_WC2(self,TfidfVec=None,**TtdifVec_kwargs):
        """
         Fit TfidfVectorizer to text and return a sparse matrix from tf-idf
        """
        if TfidfVec is None:
            # This is fit only done with the training data set.
            if len(TtdifVec_kwargs)==0:
                # Fit with default parameters
                self.tv = TfidfVectorizer(min_df=0.004,max_df=0.417)  
            else:
                # Fit with user defined parameters
                self.tv = TfidfVectorizer(**TtdifVec_kwargs)
                
            self.WC2 = self.tv.fit_transform(self.df['Text'])
        else:
            # The user passed the training TfidfVectorizer object
            self.tv = TfidfVec  
            self.WC2 = self.tv.transform(self.df['Text'])
        self.words = self.tv.get_feature_names_out()                   
 
    def get_WC2_as_df(self):
        """Converts the sparse matrix WC2 into a full DataFrame"""
        return pd.DataFrame(self.WC2.A,columns=self.words)
    def __repr__(self):
        return "MyData( nwords={} )".format(self.WC2.shape)
    def __str__(self):
        return  self.__repr__()
    
    
    def plot_confusion_matrix(y_true,y_pred,title='Confusion Matrix'):        
        labels = list(set(y_pred).union(set(y_true)))        
        cm = metrics.confusion_matrix(y_true,y_pred,labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='g',ax=ax);
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title(title); 
        ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);                             
            
train_data = MyData(train_df)
crossval_data = MyData(crossval_df,train_data.tv)
test_data = MyData(test_df,train_data.tv)
print('WC2 in training data set has size:',train_data.WC2.shape)
print('WC2 in crossval_data data set has size:',crossval_data.WC2.shape)
print('WC2 in test_data data set has size:',test_data.WC2.shape)


# In[ ]:


class MyNMF(object):
    """
    NMF model with helper methodsMyData obj
    Note: This class does not follow sklearn syntax. 
          I did refactor it but I was unable to make it work with categorical predictions,
          so all changes were undone.
    """
    def __init__(self, n_topics,**fit_params):
        """
        Inputs:
        -Xtrain_data: a MyData object
        -n_topics: number o topics to represent text
        -**fit_params: optional kwargs to pass when sklearn's NMF object is created
        """        
        self.n_topics = n_topics
        self.fit_params = fit_params        
        self.mapping_dict = None
    def fit(self,Xtrain_data,ytrain=None):
        
        # Default values
        self.fit_params['n_components'] = self.n_topics
        self.fit_params['beta_loss'] = self.fit_params.get('beta_loss',"frobenius")


# In[133]:


from sklearn import metrics
class MyNMF(object):
    """
    NMF model with helper methodsMyData obj
    Note: This class does not follow sklearn syntax. 
          I did refactor it but I was unable to make it work with categorical predictions,
          so all changes were undone.
    """
    def __init__(self, n_topics,**fit_params):
        """
        Inputs:
        -Xtrain_data: a MyData object
        -n_topics: number o topics to represent text
        -**fit_params: optional kwargs to pass when sklearn's NMF object is created
        """        
        self.n_topics = n_topics
        self.fit_params = fit_params        
        self.mapping_dict = None
    def fit(self,Xtrain_data,ytrain=None):
        
        # Default values
        self.fit_params['n_components'] = self.n_topics
        self.fit_params['beta_loss'] = self.fit_params.get('beta_loss',"frobenius")
        self.fit_params['init'] = self.fit_params.get('init',"nndsvda")
        
        # Fit NMF model and store matrix W such that WC2 = W*H
        self.model_ = NMF(**self.fit_params).fit(Xtrain_data.WC2)
        self.words_ = Xtrain_data.words
        
        
        if ytrain is None:
            # Trivial categories
            self.mapping_dict_ = {i:i for i in range(self.n_topics)}
            return self
        
        W = self.transform(Xtrain_data)
        pred_ids = self.predict_topic_ids(W)
        self.train_score_, _, _ = self.match_topic_ids_to_categories(pred_ids,ytrain)        
        return self
    
    def predict(self,Xdata):
        W = self.transform(Xdata)
        pred_ids = self.predict_topic_ids(W)
        y_pred = self.map_topic_id_to_cats(pred_ids)
        return y_pred
    def transform(self,Xdata):
        W = self.model_.transform(Xdata.WC2)
        return W     
    def predict_topic_ids(self,W):    
        """ Returns indexes of largest rows in W for NMF model. There are the predicted topic ids per article"""
        return [np.argmax(row) for row in W]
    
    def match_topic_ids_to_categories(self,pred_ids,y_true, metric=metrics.accuracy_score):   
        """        
        Try permutations to map pred_ids with the categories in y_true by maximizing a given metric
        Inputs:
        - pred_ids: vector of predicted topic ids (from matrix W)
        - y_true: true categories. (Category column in dataframe)
        - metric: desired metric function with args metric(y_true,y_pred) to be maximized. Default: accuracy_score
        Returns:
        - best_metric: achieve score
        - y_pred_best: list of predicted categories
        - best_perm: dictionary containing the mapping of unique values in pred_ids to each category
        """           
        # It could be the case that there is a different number of unique elements in pred_ids and y_true.        
        # We force the mapping to fit into the categories in y_true        
        cats = set(y_true) 
        num_cats = len(cats)                      
         # initial guess
        id_to_cat = {i:c for i,c in enumerate(cats)} # we try all permutations of the keys of this dictionary  
        
        y_pred_best = [id_to_cat[c] for c in pred_ids]        
        best_metric = metric(y_true,y_pred_best)
        best_perm = id_to_cat
        # Find a permutation to maximize given metric
        for perm_ids in permutations(id_to_cat):
            id_to_cat = {i:c for i,c in zip(perm_ids,cats)}
            y_pred = [id_to_cat[c] for c in pred_ids] 
            curr_metric = metric(y_true,y_pred)
            if curr_metric>best_metric:
                best_metric = curr_metric
                y_pred_best = y_pred
                best_perm = id_to_cat   
        self.mapping_dict_ = best_perm
        return best_metric,y_pred_best,best_perm
    
    def map_topic_id_to_cats(self,pred_ids,mapping_dict=None):
        """
        Convert column indexes in W to categories based on a mapping dictionary
        """
        if mapping_dict is None: mapping_dict = self.mapping_dict_
        return list(map(lambda ix: mapping_dict[ix],pred_ids))
    
    def get_topic_words(self,n_words):
        """"
        Get dictionary with topic id and n words per topic from matrix H in NMF
        Inputs:
        - n_words: number of words to represent each topic
        """ 
        # Matrix H: each column is a word, each row is a topic.
        """Words with highest scores are representative of a topic"""
        H = self.model_.components_ 
        
        topic_names = [self.mapping_dict_[k] for k in range(self.n_topics)]
        
        topics_dict = {topic_names[k]: list(self.words_[MyNMF.get_ind_largest(row,n_words)]) \
                       for k,row in enumerate(H)}        
        return topics_dict
    
    def print_top_words(self,n_words:int =4):        
        """Print a list of top words and the corresponding topic"""
        print('Top {} words per topic:'.format(n_words))
        topics_dict = self.get_topic_words(n_words)
        [print('words: {} => topic: {}'.format(v,k)) for k,v in topics_dict.items()];
        def __repr__(self):
            return "NMF model  with {} words and {} topics".format(len(self.words_),self.n_topics)
    def __str__(self):
            return self.model_.__str__()         
    @staticmethod
    def get_ind_largest(x,k):
        """
        Return indexes of k largest components in a 1 D numpy array.
        (Fast method)
        """        
        ind_largest= x.argpartition(-k)[-k:] # this is O(n), but unsorted.
        return ind_largest[np.argsort(-x[ind_largest])]    
    @staticmethod
    def plot_topic_words(H, words, n_words, title,word_types=None):
        """Producs a bar plot for each predicted NMF topic showing the highest n scores and their words per topic"""
        num_topics = H.shape[0]
        ncols = 5
        nrows = int(np.ceil(num_topics/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(30, 8), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(H):
            ind_largest = MyNMF.get_ind_largest(topic,n_words)
            top_words = words[ind_largest]
            weights = topic[ind_largest]
            ax = axes[topic_idx]
            ax.barh(top_words, weights, height=0.5)
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 25})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=16)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)                
            fig.suptitle(title, fontsize=30)
            
            # What is the type of the most representative word?


# In[135]:


# Although these methods are implemented in the class, 
# they are done here as a script for an easier understanding of the method.
from sklearn.decomposition import NMF
n_top_words = 4
n_topics = 5
print('*'*55)
print('  Performing NMF X = W*H with {} topics on training set'.format(n_topics))
print('*'*55)
nmf_model = NMF(n_topics,
                beta_loss='frobenius',
               init='nndsvda')
W_train =nmf_model.fit_transform(train_data.WC2)

H_train = nmf_model.components_
print('The chosen latent dimension p is:',n_topics)
print('The size of W for training set is:',W_train.shape)
print('The size of H for training set is:',H_train.shape)
print('=>W train:\n',W_train)
print('=>H train:\n',H_train)


# In[136]:


print("H train contains coefficients to p={} rows (topics), each defined by a combination of {} words.".format(*H_train.shape))
print("We will represent each topic (row) with the {} words that have the largest coefficient in H".format(n_top_words))

topics = {i:list(train_data.words[MyNMF.get_ind_largest(row,n_top_words)])  for i,row in enumerate(H_train)}
print(topics)

for i,row in enumerate(H_train):
    ind_largest = MyNMF.get_ind_largest(row,n_top_words)
    print('\nTopic in Row {} is represented by the words: \n{} with coefficients:\n{}'.format(
        i,train_data.words[ind_largest],np.round(row[ind_largest],2)))
print('\nCompare these words to the official categories:\n',set(train_data.df['Category']))


# In[137]:


MyNMF.plot_topic_words(H_train,train_data.words,n_top_words,
               "Topics in NMF model (Frobenius norm)")


# In[138]:


print('Training set with category and predicted topics:')
t_df= train_df.copy()
t_df['predicted_topics']=[topics[np.argmax(article)] for article in W_train]
t_df['predicted_topic_id']=[np.argmax(article) for article in W_train]
t_df['cat_id']=t_df['Category'].factorize()[0]
t_df


# In[140]:


# Repeat the above calculations using class methods from MyNMF:
from itertools import permutations 
y_true_train = train_data.df['Category']
nmf_model = MyNMF(n_topics=5).fit(train_data,y_true_train)
y_pred_train = nmf_model.predict(train_data)
acc = metrics.accuracy_score(y_true_train,y_pred_train)
nmf_model.print_top_words(10)

print('\nNMF predictions for the training set. (Accuracy = {:.3f}%)'.format(acc*100))
t_df= train_df.copy()
t_df['predicted Category (y_pred)']=y_pred_train
t_df


# In[141]:


#Compute accuracy and confusion matrix.
print('\nAccuracy for training set based on NMF: {:.3f}%'.format(acc*100))
MyData.plot_confusion_matrix(y_true_train,y_pred_train,'Confusion matrix for training set based on NMF')


# In[142]:


# Evaluate model with cross validation data
y_pred_cv = nmf_model.predict(crossval_data)
y_true_cv = crossval_data.df['Category']
acc = metrics.accuracy_score(y_true_cv,y_pred_cv)
print('\nAccuracy for cross val set based on NMF: {:.3f}%'.format(acc*100))
MyData.plot_confusion_matrix(y_true_cv,y_pred_cv,'Confusion matrix for cross val set based on NMF')


# In[143]:


# Predict categories
y_pred_test = nmf_model.predict(test_data)
t_df = test_data.df.copy()
t_df['predicted Category']=y_pred_test
print('Predicted categories on testing set:')
t_df


# In[144]:


# DataFrame for submission
nmf_predictions = test_data.df.copy()
nmf_predictions['Category']=y_pred_test
nmf_predictions.drop("Text",axis=1,inplace=True)
nmf_predictions


# In[146]:


# Repeat model fit including the full data set
train_df_full = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Train.csv").drop_duplicates(subset=['Text'])
train_data_full = MyData(train_df_full)
y_true_train = train_data_full.df['Category']

nmf_model = MyNMF(5).fit(train_data_full,y_true_train)
y_pred_train = nmf_model.predict(train_data_full)
acc = metrics.accuracy_score(y_true_train,y_pred_train)
print('***Model trained with full training set:***')
nmf_model.print_top_words(10)
print('\nAccuracy for full training+cross val set based on NMF trained with full data set: {:.3f}%'.format(acc*100))
MyData.plot_confusion_matrix(y_true_train,y_pred_train,'Confusion matrix for train+cross val sets based on  NMF model')


# In[148]:


import sys
# Delete large variables to save space
local_vars = np.array(list(locals().items()),dtype=object)
var_sizes = np.array([sys.getsizeof(obj) for var, obj in local_vars])
size_threshold = 10000 # bytes
ind = np.argsort(-var_sizes)
local_vars = local_vars[ind]
var_sizes = var_sizes[ind]
print('Large variables to be deleted:')
[print('name:',v[0],' =>size:',s,'bytes') for v,s in zip(local_vars,var_sizes) if s>10000 and v[0][0]!='_'];


# In[150]:


# delete
import gc
print('Deleting variables..')
for v,s in zip(local_vars,var_sizes):
    if s>size_threshold and v[0][0]!='_':
        try: 
            exec('del '+ v[0])
        except NameError:
            pass
gc.collect();
print('...Done.')


# In[165]:


from itertools import product
from time import perf_counter
from tqdm import tqdm
def product_dict(**kwargs):
    """" Build iterator for grid search with all combinations from lists in a dictionary"""
    keys = kwargs.keys()
    vals = kwargs.values()
    return [dict(zip(keys, instance)) for instance in product(*vals)]

def NMF_gridsearch(train_df,crossval_df,NMF_all_params:dict,TtdifVec_all_params:dict,verbose=1,min_acc_threshold:float=0.8):
    """
    - min_acc_threshold: float between 0.0 and 1.0. Keep only models with accuracy greater than or equal to this value.
    """
    assert 0.0<=min_acc_threshold<=1.0
      
    # Get list
    NMF_all_params=product_dict(**NMF_all_params)
    TtdifVec_all_params=product_dict(**TtdifVec_all_params)
    num_models =len(NMF_all_params)*len(TtdifVec_all_params)    
    cont = 0
    best_val_acc = 0.0    
    ALL_RESULTS = []
    if verbose in [1,2]:
        print('Grid search with {} model hyperparameter combinations'.format(num_models))
    if verbose==1:
        print('Showing only models with train and val. acc>={:.3f}%'.format(min_acc_threshold*100))
    elif verbose==2:
        print('Showing models under test. Acc=-1 meand a ValueError was raised.')        
    t0 = perf_counter()    
    for NMF_params,TtdifVec_params in tqdm(product(NMF_all_params,TtdifVec_all_params),total=num_models):
            cont+=1     
            results = test_model_params(train_df,TtdifVec_params,NMF_params,crossval_df)               
            train_acc = results['train_acc']
            val_acc = results['val_acc']
            is_high_acc_model = min_acc_threshold<val_acc and min_acc_threshold<train_acc
            
            if val_acc>best_val_acc:
                best_val_acc = val_acc
                best_results = results
            
            if is_high_acc_model:
                ALL_RESULTS.append(results)
            if verbose == 2 or (is_high_acc_model and verbose==1):
                e_time = (perf_counter() - t0)/60
                print('\nModel {} of {} (e_time={:.1f}min):'.format(cont,num_models,e_time))
                print(TtdifVec_params)
                print(NMF_params)
                print(' Train acc:{:.3f}%, Validation acc:{:.3f}%, (best val acc:{:.3f}%)'.format(
                      train_acc*100,val_acc*100,best_val_acc*100))
    if verbose==1:
        print('{} models found with val_acc>{:.3f}% '.format(len(ALL_RESULTS),100*min_acc_threshold))
    return best_results,ALL_RESULTS    
    
def test_model_params(df_train,TtdifVec_params,NMF_params,df_crossval=None):
    """ Test one set of model parameters (Allows grid search in our NMF pipeline)"""    
    # Train model
    
    train_data = MyData(df_train,TfidfVec=None,clean_text=True,**TtdifVec_params)
    y_true_train = train_data.df['Category']
    try:
        nmf_model = MyNMF(5,**NMF_params).fit(train_data,y_true_train)
    except ValueError:
        # some parameter combinations are invalid     
        model_results = {'train_acc':-1.0,
                         'val_acc':-1.0,
                         'train_data':None,
                         'nmf_model':None,
                         'TtdifVec_params':TtdifVec_params,
                         'NMF_params':NMF_params}
        return model_results
     # Evaluate model with cross validation data
    if df_crossval is None:
        val_acc = None
    else:        
        cv_data = MyData(df_crossval,TfidfVec=train_data.tv)
        y_pred_cv = nmf_model.predict(cv_data)
        y_true_cv = cv_data.df['Category']
        val_acc = metrics.accuracy_score(y_true_cv,y_pred_cv)
    
    model_results = {'train_acc': nmf_model.train_score_ ,
         'val_acc': val_acc,
         'train_data': train_data,
         'nmf_model': nmf_model,
          'TtdifVec_params': TtdifVec_params,
          'NMF_params': NMF_params
        }
    return model_results
def retrain_models(ALL_MODELS:list,df_train):
    return [retrain_model(m,df_train) for m in ALL_MODELS]

def retrain_model(nmf_model_dict,df_train):
    return test_model_params(df_train,df_crossval=None,
                      TtdifVec_params = nmf_model_dict['TtdifVec_params'],
                      NMF_params = nmf_model_dict['NMF_params'])
    
def bagging_nmf_prediction(ALL_MODELS,X_df):
    ypreds = [predict_nmf_from_result(m,X_df) for m in  ALL_MODELS]
    # Majority voting: return mode
    ypred =  pd.DataFrame(ypreds).mode().values[0].tolist()
    return ypred
def predict_nmf_from_result(nmf_model_dict,X_df):
    nmf_model = nmf_model_dict['nmf_model']
    Xdata = MyData(X_df,TfidfVec=nmf_model_dict['train_data'].tv)    
    y_pred = nmf_model.predict(Xdata)        
    return y_pred
def filter_unique_model_results(ALL_MODELS:list):
    # Models with same training and validation accuracy scores are filtered to leave only one of each.
    return pd.DataFrame(ALL_MODELS).drop_duplicates(subset=['train_acc','val_acc']).to_dict('records')


# In[158]:


# Reload data
print('Reload data. \n Note: the same train/cross val split is generated to allow direct comparison of model performance.')
train_df_full = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Train.csv").drop_duplicates(subset=['Text'])
test_df = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Test.csv")
train_df, crossval_df = train_test_split(train_df_full,stratify=train_df_full[['Category']],test_size=0.2,random_state = 310)
print('Training set:')
train_df


# In[166]:


# NMF Hyperparameter space.

# # Original search space
# NMF_all_params= {
#     'init' : ['random','random', 'nndsvda', 'nndsvdar','nndsvd'], ## nndsvd leads to low accuracy
#     'solver' : ['mu','cd'], # 'cd always leads to lower accuracy    
#     'beta_loss' : ['kullback-leibler','frobenius', 'itakura-saito'], # ['frobenius', 'itakura-saito'] both lead to low accuracy
#     'alpha_W' : [0.0,0.01,0.1]
#     }
# Reduced search space
NMF_all_params= {
    'init' : ['nndsvda','nndsvdar'], ## nndsvd leads to low accuracy
    'solver' : ['mu'], # 'cd' leads to low accuracy    
    'beta_loss' : ['kullback-leibler'], # ['frobenius', 'itakura-saito'] lead to low accuracy
    'alpha_W' : [0.0,0.01,0.05,0.1]
    }


# tf-idf Hyperparameter space.

# #Original search space
# TtdifVec_all_params = {
#     'stop_words' : ['english',None],
#     'ngram_range' : [(1,1),(1,2)],
#     'max_df' : [0.5,0.6, 0.7,0.8,0.9,0.95,0.97,0.98,0.99],
#     'min_df' : [0.001,0.002,0.005,0.02, 0.05,0.1]
#     }
# Reduced search space
TtdifVec_all_params = {
    'stop_words' : ['english'],
    'ngram_range' : [(1,2)],
    'max_df' : [0.5,0.7,0.9],
    'min_df' : [0.002]
    }

best_NMF_model,ALL_MODELS = NMF_gridsearch(train_df,crossval_df,NMF_all_params,TtdifVec_all_params,verbose=1,min_acc_threshold=0.965);
print('\nBest model is:\n',best_NMF_model)
print('\nList of models with acceptable accuracy:')
pd.DataFrame(ALL_MODELS)


# In[167]:


print('Parameters of the best individual model:')
best_NMF_model


# In[168]:


y_pred_best_model = predict_nmf_from_result(best_NMF_model,train_df_full)
y_true = train_df_full['Category']
acc_best_model = metrics.accuracy_score(y_true,y_pred_best_model)
print('The accuracy of the best individual NMF model evaluated on full train + cross val set is: {:.3f}%'.format(acc_best_model*100))
MyData.plot_confusion_matrix(y_true,y_pred_best_model,'Confusion matrix for cross val set based on best individual NMF')


# In[169]:


MyNMF.plot_topic_words(H=best_NMF_model['nmf_model'].model_.components_,
                       words = best_NMF_model['nmf_model'].words_,
                       n_words = 10,
                       title = "Topics in best NMF model (after grid search)")


# In[170]:


best_NMF_model_retrained = retrain_model(best_NMF_model,train_df_full)
y_pred_best_model = predict_nmf_from_result(best_NMF_model_retrained,train_df_full)
acc_best_model = metrics.accuracy_score(y_true,y_pred_best_model)
print('The training accuracy of the NMF model with best hyperparameters retrained on full train + cross val set is: {:.3f}%'.format(acc_best_model*100))
MyData.plot_confusion_matrix(y_true,y_pred_best_model,'Confusion matrix for nmf model with best hyperparameters retrained on full dataset')


# In[171]:


print('List of different (unique) models to use as a bag of models:')
ALL_MODELS = filter_unique_model_results(ALL_MODELS)
pd.DataFrame(ALL_MODELS)


# In[172]:


y_pred_bag = bagging_nmf_prediction(ALL_MODELS,train_df_full)
acc_bag = metrics.accuracy_score(y_true,y_pred_bag)
print('The accuracy of a bag with {} models on full train+cross val set using majority voting is: {:.3f}%'.format(len(ALL_MODELS),acc_bag*100))
MyData.plot_confusion_matrix(y_true,y_pred_bag,'Confusion matrix for full set based on bagging of {} NMF models'.format(len(ALL_MODELS)))


# In[173]:


y_pred_test = bagging_nmf_prediction(ALL_MODELS,test_df)

print('Since bagging reduces variance, we choose bagging. Our submission is based on a bagging of  {} NMF models'.format(len(ALL_MODELS)))
    
# DataFrame for submission
filename = 'submission.csv'
try:
    os.remove("/kaggle/working/submission.csv") # remove csv file if it exists
except: 
    pass
nmd_predictions = test_data.df.copy()
nmd_predictions['Category']=y_pred_test
nmd_predictions.drop("Text",axis=1,inplace=True)
print('Submission test set predictions with NMF saved as {}'.format(filename))
nmd_predictions.to_csv('submission.csv', index=False)
nmd_predictions


# In[174]:


print('The following Hyperparameters of TfidfVectorizer were found to produce best result for NMF models, and will be used to pre-process the inputs of the supervised model:')
TtdifVec_params = best_NMF_model['TtdifVec_params']
print(TtdifVec_params)
Xdata = MyData(train_df,TfidfVec=None,clean_text=True,**TtdifVec_params)


# In[175]:


Xtrain = Xdata.WC2
ytrain = train_df['Category']
s = 1 - WC2.count_nonzero()/np.prod(WC2.shape)
print('\nWC2 will be named Xtrain. This is a sparse matrix of size {} with sparsity = {:.2f}%'.format(Xtrain.shape,100*s))
print("\nBased on WC2, we will be training a model with {} inputs and {}  samples".format(Xtrain.shape[1],Xtrain.shape[0]))


# In[178]:


from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
rf_model=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
rf_model.fit(Xtrain,ytrain)
# Cross validaton set
Xcv = MyData(crossval_df,TfidfVec=Xdata.tv,clean_text=True,**TtdifVec_params).WC2
y_pred_cv=rf_model.predict(Xcv)
y_true_cv = crossval_df['Category']
rf_cv_acc = metrics.accuracy_score(y_true_cv,y_pred_cv)
print('The accuracy of our first random forest classifier on the cross validation set is {:.2f}%'.format(rf_cv_acc*100))
MyData.plot_confusion_matrix(y_true_cv,y_pred_cv,"Confusion matrix of RF model on cross validation set")


# In[179]:


print('Number of features in input data:',WC2.shape[1])
print('log2(N_features): {:.1f}'.format(np.log2(WC2.shape[1])))
print('sqrt(N_features): {:.1f}'.format(np.sqrt(WC2.shape[1])))


# In[181]:


from sklearn.model_selection import GridSearchCV
Xtrain_full = MyData(train_df_full,TfidfVec=Xdata.tv,clean_text=True,**TtdifVec_params).WC2
ytrain_full = train_df_full['Category']

# Create a Random Forest Classifier
clf=RandomForestClassifier(n_jobs=-1)

# Hyperparameters (the original search space was larger)
parameters = {
    'n_estimators' : [100,250],
    'criterion' : ('gini','entropy'),
    'max_depth' : (None,50,100),
    'max_features' : ('sqrt','log2',50),
    }
# Grid search
print('Tuning hyperparameters')
clf2 = GridSearchCV(clf, parameters, verbose = 3)
clf2.fit(Xtrain_full,ytrain_full)


# In[182]:


print("Highest accuracy for RF hyperparameter tuning is: {:.3f}%".format(clf2.best_score_ ))
R = pd.DataFrame(clf2.cv_results_)
print("Showing top 10 models")
R = R.sort_values(by="rank_test_score",axis=0)
R.head(10)


# In[183]:


# Filter best simple model
# R =  R[(R['rank_test_score']<=10) & (R['param_max_features']=='log2')].iloc[0]
R =  R.iloc[0]
print('Parameters for the chosen model, with accuracy = {:.3f}%:'.format(R['mean_test_score']*100))
best_RF_model_params=R['params']
best_RF_model_params


# In[184]:


# Retrain model
rf_model = RandomForestClassifier(n_jobs=-1,**best_RF_model_params)
rf_model.fit(Xtrain,ytrain)
y_pred_cv=rf_model.predict(Xcv)
y_true_cv = crossval_df['Category']
rf_cv_acc = metrics.accuracy_score(y_true_cv,y_pred_cv)
print('The accuracy of tuned random forest classifier on the cross validation set is {:.3f}%'.format(rf_cv_acc*100))
MyData.plot_confusion_matrix(y_true_cv,y_pred_cv,"Confusion matrix of RF model on cross validation set")


# In[185]:


rf_model.feature_importances_
forest_importances = pd.Series(rf_model.feature_importances_, index=Xdata.words,name='importance')
forest_importances.sort_values(axis=0,ascending=False,inplace=True)
fig, axs = plt.subplots(figsize=(15,5))
sns.barplot(x=forest_importances.head(10).index,y=forest_importances.head(10).values,palette='colorblind',ax=axs);
axs.set_xlabel('Word')
axs.set_ylabel('Importance')
axs.set_title('Classification importance of top 10 words in random forest model')
# start_time = time.time()
# result = permutation_importance(
#     forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
# elapsed_time = time.time() - start_time
# print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# forest_importances = pd.Series(result.importances_mean, index=feature_names


# In[187]:


# learning_curve
from sklearn.model_selection import learning_curve
print('Computing learning curve for Random Forest model...\n')
rf_model = RandomForestClassifier(n_jobs=-1,**best_RF_model_params)
train_sizes, train_scores, test_scores = learning_curve(
        rf_model,
        Xtrain_full,
        ytrain_full,
        verbose=3)


# In[188]:


def plot_learning_curve(train_sizes,train_scores,test_scores,fig_title,axes=None):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if axes is None:fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    
    axes.errorbar(train_sizes,train_scores_mean,1.96*train_scores_std,fmt="o-",label="Training acc");
    axes.errorbar(train_sizes,test_scores_mean,1.96*test_scores_std,fmt="o-",label="Cross-validation acc");
    axes.legend(loc=4);
    axes.set_xlabel('Training size');
    axes.set_ylabel('Accuracy');
    axes.grid()
    axes.set_title(fig_title)
    


# In[189]:


# Manually compute training curve for NMF model for same number of train sizes
train_scores_nmf = np.zeros((5,len(train_sizes)))
test_scores_nmf = np.zeros((5,len(train_sizes)))

train_df, crossval_df = train_test_split(train_df_full,stratify=train_df_full[['Category']],test_size=0.2,random_state = 310)

for  i,s in tqdm(enumerate(train_sizes),desc='Computing NMF model training curve with 5-fold validaton',total=len(train_sizes)): 
    for j in range(5):
        try:
            train_df_sub, _ = train_test_split(train_df,stratify=train_df[['Category']],train_size=s)        
        except ValueError:
            train_df_sub = train_df
            
        results = test_model_params(train_df_sub,
                                    best_NMF_model['TtdifVec_params'],
                                    best_NMF_model['NMF_params'],crossval_df)
        train_scores_nmf[i,j] = results['train_acc']
        test_scores_nmf[i,j] =results['val_acc']


# In[190]:


# Plot learning curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5),sharey=True)
plot_learning_curve(train_sizes,train_scores,test_scores,'Learning curve for Random Forest model',axes[0])
plot_learning_curve(train_sizes,train_scores_nmf,test_scores_nmf,'Learning curve for NMF model',axes[1])


# In[211]:


# Methods to recombine text
import functools
from itertools import combinations
def get_word_cum_sum_split_by_points(s):
    return np.cumsum(np.array([len(q.split()) for q in s.split('.')]))

def split_text_by_quantile(text,quantile='random'):   
    """
    Inputs:
    - Text: a string including some periods ('.') to be split at some quantile
    - quantile: float between 0.0 and 1.0 or 'random', indicating the
            quantile of number of words to be used as a split base.
            If 'random' is chosen, a random value is choseon from [0.25, 0.50, 0.75]
    """    

        
    word_lists_per_sentence = [sentence.split() for sentence in text.split('.') if len(sentence) is not 0]
    cum_num_words_per_sentence = np.cumsum(np.array([len(sentence) for sentence in word_lists_per_sentence]))
    
    try:
        q = np.quantile(cum_num_words_per_sentence,quantile)
    except TypeError: # quantile = 'random'
        q = np.quantile(cum_num_words_per_sentence,np.random.choice([0.25,0.5,0.75]))
        
    cut_point = find_nearest_ind(cum_num_words_per_sentence,q)
    add_sentences = lambda x,y: x + y
    add_words = lambda x,y: x +' '+ y
    n=cum_num_words_per_sentence[-1]

    part1 = functools.reduce(add_words, functools.reduce(add_sentences, word_lists_per_sentence[0:cut_point]))
    part2 = functools.reduce(add_words, functools.reduce(add_sentences, word_lists_per_sentence[cut_point:]))

    return part1, part2

def recombine_texts(text1,text2,quantile=0.5):        
    t1_part1, t1_part2 = split_text_by_quantile(text1,quantile)
    t2_part1, t2_part2 = split_text_by_quantile(text2,quantile)
    new_text1 = t1_part1 + ' ' + t2_part2
    new_text2 = t2_part1 + ' ' + t1_part2
    return new_text1,new_text2

def create_combinations_df(Texts:pd.Series,combinations_dict:dict,limit_per_category='min'):
    """
    Combine texts to create a DataFrame with 'Text'  and 'Category' that can be used to augment training size
    
    """
    assert (isinstance(limit_per_category ,int) or limit_per_category in ['min',None]), "limit should be None,'min', or an integer"
    if limit_per_category is None:
        limit = np.inf
    elif limit_per_category is 'min':
        limit = min([v['num_pairs'] for v in combinations_dict.values()])
    else:
        limit = limit_per_category
    results = {'Category':[],
               'Text':[]}
    print('\nComputing 2*{} text recombinations per category:'.format(limit))
    for cat, v in combinations_dict.items():        
        n_items = min([v['num_pairs'],limit])
        for k in tqdm(range(n_items),desc='Category = {}'.format(cat)):
            i,j = v['pairs'][k]
            t1, t2 =recombine_texts(Texts[i],Texts[j])
            results['Category'].append(cat)
            results['Category'].append(cat)
            results['Text'].append(t1)
            results['Text'].append(t2)
    return pd.DataFrame(results)
            
def get_combination_indexes(train_df):
    """
    Get indexes with all pairs of Clean texts from same categories.
    Returns a dictionary with key=Category, where each value is a dictionaty containing:
    - num_pairs: number of pairs that can be generated
    - pairs: 
    """
    G = train_df.groupby('Category')
    combinations_dict = dict()
    for category,g in G:
        ind = g.index.values
        np.random.shuffle(ind) # this will only partially randomize
        n= len(ind)
        combinations_dict[category] = {'num_pairs':int((n*(n-1))/2),
                            'pairs': create_shuffled_pairs(ind)
                           }    
    return combinations_dict

def create_shuffled_pairs(x):
    pairs = [_ for _ in combinations(x,2)]
    np.random.shuffle(pairs)
    return pairs
        
def find_nearest_ind(a,a0):
    """ find index of closest value in array a to a0"""
    return np.abs(a - a0).argmin()


# In[195]:


# Reload data and remove duplicates.
train_df_full = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Train.csv").drop_duplicates(subset=['Text'])
train_df, crossval_df = train_test_split(train_df_full,stratify=train_df_full[['Category']],test_size=0.2,random_state = 310)
crossval_df['Text'] = clean_text(crossval_df['Text'])
train_df_full['Text'] = clean_text(crossval_df['Text'])

train_df['Text_with_dots']=clean_text(train_df,keep_dots=True)
train_df


# In[196]:


def count_dots(text):
    return text.count('.')

# Statistics about the number of dots in articles
train_df['Dot count'] = train_df['Text_with_dots'].apply(count_dots)

fig,axs = plt.subplots(nrows=2,figsize=(15,10),sharex=True)
sns.histplot(train_df,
             ax=axs[0],
             x='Dot count',
             hue='Category',
             palette='colorblind').set(
             title='Number of dots in text per cateory (training set)');

sns.boxplot(        
        ax = axs[1],
        data = train_df, 
        x = 'Dot count', 
        y = 'Category',
        palette = 'colorblind',    
        ).set(
            title = 'Number of dots in text per cateory (training set)');

print('Descriptive statictics of the number of periods per article in the training set:')
pd.DataFrame(train_df['Dot count'].describe())


# In[197]:


print('*** Original text of article with most dots: ***')
train_df.loc[train_df['Dot count'].idxmax(),'Text']


# In[206]:


q = 0.25
print(' Split 2 sample texts by paragraph according to quantile q{:.0f}:'.format(q*100))
print('\nText 1 ():')
T1 = train_df[train_df['Category']=='business']['Text_with_dots'].iloc[0]
for i,t in enumerate(split_text_by_quantile(T1,q)):
    print(' Part {} ({} words):\n -> {}'.format(i+1,len(t.split(' ')),t))
    
print('\nText 2')
T2 = train_df[train_df['Category']=='business']['Text_with_dots'].iloc[1]
for i,t in enumerate(split_text_by_quantile(T2,q)):
    print(' Part {} ({} words):\n -> {}'.format(i+1,len(t.split(' ')),t))


# In[207]:


print("Recombined texts:")
recombine_texts(T1,T2,quantile=0.25)


# In[208]:


col1 = 'original articles (n)'
col2 = 'possible new articles by recombination (p)'
article_counts = pd.DataFrame(train_df.groupby('Category').count()['Text'].rename(col1))
article_counts['%n']=np.round(100*article_counts[col1]/sum(article_counts[col1]),2)
article_counts[col2] = article_counts[col1].apply(lambda n: n*(n-1))
article_counts['%p']=np.round(100*article_counts[col2]/sum(article_counts[col2]),2)
article_counts


# In[209]:


print('Ratio of samples in class with most samples to samples in class with least number of samples:')
r1 = np.round(article_counts[col1].max()/article_counts[col1].min(),2)
r2 = np.round(article_counts[col2].max()/article_counts[col2].min(),2)
print('=> ',col1,' ratio =',r1)
print('=> ',col2,' ratio =',r2)


# In[212]:


train_df['Text_with_points']=clean_text(train_df,keep_dots=True)
new_samples_df = create_combinations_df(train_df['Text_with_points'],get_combination_indexes(train_df),limit_per_category=460)
train_df['Text']=clean_text(train_df['Text'])
train_df=train_df.loc[:,['Text','Category']] # filter desired columns
train_df_aug = pd.concat([train_df,new_samples_df])


# In[213]:


print('Original training set with {} samples:' .format(len(train_df)))
train_df.groupby('Category').describe()


# In[214]:


print('Augmented training set with {} samples (may includes some repeated samples):'.format(len(train_df_aug)))
train_df_aug.groupby('Category').describe()


# In[215]:


train_df_aug.drop_duplicates(subset=['Text'],inplace=True)
print('After removing duplicates, we get: {} samples'.format(len(train_df_aug)))
print('Data augmentation has incresed the training data by a factor of: {:.0f}X'.format(len(train_df_aug)/len(train_df)))


# In[216]:


_,axs = plt.subplots(ncols=1,figsize=(8,5))
w1 = train_df['Text'].apply(lambda s: len(s.split(' ')))
w2 = train_df_aug['Text'].apply(lambda s: len(s.split(' ')))
sns.histplot(w1,color='blue',alpha=0.3,label='Original training data',kde=True,ax=axs)
sns.histplot(w2,color='red',alpha=0.3,label='Augmented traing data',kde=True,ax=axs).set(xlabel='Words per article')
axs.legend();
pd.DataFrame({'words per article in original set':w1.describe(),'words per article in augmented set':w2.describe()})


# In[217]:


print('This computation may take around 10 minutes...')
fracs = [0.15, 0.33, 0.55, 0.78, 0.85, 1.0]
nsamples = []

# Manually compute training curve for NMF model for same number of train sizes
train_scores_nmf = np.zeros((len(fracs),5))
test_scores_nmf = np.zeros((len(fracs),5))
train_scores_rf = np.zeros((len(fracs),5))
test_scores_rf = np.zeros((len(fracs),5))


for  i,f in tqdm(enumerate(fracs),desc='Computing training curves for NMF and RF models using augmented dataset with 5-fold validaton',total=len(fracs)): 
    for j in range(5):        
        try:
            train_df_aug_sample, _ = train_test_split(train_df_aug,stratify=train_df_aug[['Category']],train_size=f)        
        except ValueError:
            train_df_aug_sample = train_df_aug                    
        
        # Prepare data
        train_data_aug = MyData(train_df_aug_sample,TfidfVec=None,clean_text=False,**best_NMF_model['TtdifVec_params'])
        cross_val_data = MyData(crossval_df,TfidfVec=train_data_aug.tv,clean_text=True)
        y_true_train = train_data_aug.df['Category']
        y_true_cv = cross_val_data.df['Category']
# NMF model
        mnf = MyNMF(5, **best_NMF_model['NMF_params']).fit(train_data_aug, y_true_train) 
        train_scores_nmf[i,j] = mnf.train_score_
        test_scores_nmf[i,j] = metrics.accuracy_score(y_true_cv, mnf.predict(cross_val_data))
        
        # RF model
        rf_model = RandomForestClassifier(n_jobs=-1,**best_RF_model_params).fit(train_data_aug.WC2, y_true_train)
        train_scores_rf[i,j] = metrics.accuracy_score(y_true_train, rf_model.predict(train_data_aug.WC2))
        test_scores_rf[i,j] = metrics.accuracy_score(y_true_cv, rf_model.predict(cross_val_data.WC2))
    nsamples.append(len(train_df_aug_sample))


# In[218]:


# Plot learning curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5),sharey=True)
axes[0].axvline(x = len(train_df),color = 'r',ls='--', label = 'Original size of training set');
axes[1].axvline(x = len(train_df),color = 'r',ls='--', label = 'Original size of training set');
axes[0].axhline(y = 0.97,color = 'gray',ls='--', label = '97% acc');
axes[1].axhline(y = 0.97,color = 'gray',ls='--', label = '97% acc');
plot_learning_curve(nsamples,train_scores_rf,test_scores_rf,'Learning curve for Random Forest model using augmented data',axes[0]);
plot_learning_curve(nsamples,train_scores_nmf,test_scores_nmf,'Learning curve for NMF model using augmented data',axes[1]);
ymin,_ = axes[0].get_ylim()
axes[0].set_ylim(ymin,1.01);
axes[1].set_ylim(ymin,1.01);


# In[221]:


test_df = pd.read_csv("/Users/pranavchole/Downloads/learn-ai-bbc/BBC News Test.csv")
test_data = MyData(test_df,TfidfVec=train_data_aug.tv,clean_text=True)

# Use the last MNF model in the training curve
y_pred_test = mnf.predict(test_data)

print('This new submission is based on a single NMF model trained with data augmentation')
    
# DataFrame for submission
filename = 'submission.csv'
try:
    os.remove("/Users/pranavchole/Downloads/learn-ai-bbc/submission.csv") # remove csv file if it exists
except: 
    pass
nmd_predictions = test_data.df.copy()
nmd_predictions['Category']=y_pred_test
nmd_predictions.drop("Text",axis=1,inplace=True)
print('Submission test set predictions with NMF saved as {}'.format(filename))
nmd_predictions.to_csv('/Users/pranavchole/Downloads/learn-ai-bbc/submission.csv', index=False)
nmd_predictions

