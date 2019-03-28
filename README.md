# BERT classifier <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2000px-Airbnb_Logo_B%C3%A9lo.svg.png" width="150">
###### creating a text classifier with BERT to predict ratings of Airbnb listing descriptions

Notes
------
The full length code is available in the accompanying notebook and should be viewed on [nbviewer](https://nbviewer.jupyter.org/github/stevhliu/airbnbListingDescriptions/blob/master/airbnbListingDescriptions.ipynb). A slide deck is also included for stakeholders seeking to quickly grasp the data driven insights derived from this project. Finally, an explanatory article regarding this project can be found here. 

Check out the interactive companion site [here](https://textexplorer.herokuapp.com/) (give it a few minutes to load!).

Abstract
------
Airbnb is an online community marketplace for people to book a complete travel experience, including accommodations and locally curated activities. One of the first interactions a guest has with a home is the listing description. The listing refers to the description of the home, and provides an opportunity to highlight unique features. As it turns out, the listing description occupies a decisive role in how guests perceive the overall quality of a home. It is so important that homes qualifying for the Airbnb Plus program (a collection of homes verified for quality, comfort and style) have their listing descriptions written by professional writers trained in the Airbnb Plus editorial style. But writing a unique and captivating listing description, that stands out from other homes can be difficult. Airbnb offers resources for writing a great listing description on their blog, and also includes tips on how to make a listing competitive.

However, this can be difficult to implement in practice for many hosts, who have no way of receiving feedback until after the guest leaves a rating. To help hosts maximize the probability of earning a high rating, I propose a text classification system that will allow hosts to receive instant feedback on their listing descriptions. The classifier is built on top of Google's state-of-the-art pre-trained model, BERT, a bidirectional transformer model. BERT is powerful because it understands the contextual information to the left and right of a word, giving it a deeper understanding of the language model. BERTs outputs were fed into a deep feedforward neural network with a dropout layer (to enforce regularization) and a softmax classifier. Trained on roughly 7,000 listing descriptions in San Francisco, the model achieved 81.3% accuracy in classifying the star rating a host will receive based on their listing description. In addition, the data showed that guests value private homes above corporate properties, and prefer listing descriptions that emphasize the unique qualities of a home rather than a list of generic amenities. Accordingly, hosts are recommended to focus on these unique characteristics in their descriptions, and include words that evoke homey feelings.

Insights
------
#### **Characteristic terms and associations**
Currently, there aren't very many good options for text visualization. Word clouds are perhaps the most common method, but they can be hard to interpret. It can be difficult to compare the sizes of two non-horizontally adjacent words, and longer words can appear to have an outsized impact simply because they occupy more space.

To avoid these pitfalls, Scattertext is a visualization tool that presents a scatterplot, where each axis corresponds to the frequency of a term and its related category. The term associations are determined by their scaled F-score, the harmonic mean of precision and frequency. The scaled F-score takes the normal cumulative density function of the precision and frequency scores, to scale and standardize both scores. Terms that are associated will have both high category precision and frequency, producing a higher scaled F-score.

##### **Plot guide**
* Terms used commonly in all listing descriptions are located in the upper-right corner of the plot (home, house, kitchen, apartment, room). These words are all common elements of a living space regardless of high or low star rating. On the other hand, terms used infrequently in all listing descriptions are located in the lower-left corner of the plot. For example, children is rarely used, which suggests homes accommodating children are relatively rare.
* It gets more interesting when we look at the upper-left and bottom-right corners. These are terms that are most commonly associated with high star ratings and all other ratings respectively. The colors also help identify word association. Those terms that are more aligned with high star ratings are blue, and those more associated with all other ratings are red.
* Scattertext is an interactive plot, and clicking on a word brings up an excerpt from where it was used in the listing description. There is also a search function if you want to look up a specific word. Finally, Scattertext is unable to compare multiple groups at once, which is why we need to take a one-to-many visualization approach.

```python
# import libraries for handling text
import scattertext as st
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from pprint import pprint

# load English model
nlp = spacy.load('en')

# create a scattertext corpus & look for differences between listing descriptions in ratings
# remove stop words
corpus = (st.CorpusFromPandas(listing,
                              category_col='review_scores_rating',
                              text_col='description',
                              nlp=nlp)
          .build()
          .remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True))

# generate scatterplot and save to HTML file
# set a minimum term frequency to filter out infrequent words
html = st.produce_scattertext_explorer(corpus,
          category='4.0-5.0',
          category_name='Star Rating 4.0 - 5.0',
          not_category_name='All Other Ratings',
          width_in_pixels=1000,
          minimum_term_frequency=200)
open('term-associations.html', 'wb').write(html.encode('utf-8'))
```
<p align="center">
  <img width="1000" src="https://github.com/stevhliu/airbnbListingDescriptions/blob/master/imgs/termAssociations.png" />
</p>

#### **Topic signals and categories**
It is helpful to understand what types of topics are being mentioned in listing descriptions that correspond with high star ratings. Hosts can model and incorporate these topic categories into their own listing descriptions, while learning which ones to avoid.

Empath allows us to learn the topic signals within listing descriptions, by generating and validating categories from a few seed terms. For example, the terms vintage, trendy and fashion would fall under the hipster category in Empath. Empath works by learning the word representations from 1.8 billion words from fiction stories with a neural network, and creates a vector space that measures the cosine similarity between words. Given a set of new seed terms, Empath utilizes its learned neural weights to discover related terms, and then generates a new lexical category.

Integrating Empath with Scattertext, we can build a scatterplot of topics associated with high star ratings and all other ratings.

```python
# initialize Empath topics/categories
feat_builder = st.FeatsFromOnlyEmpath()

# create corpus from Empath
empath_corpus = (st.CorpusFromParsedDocuments(listing,
                                             category_col='review_scores_rating',
                                             feats_from_spacy_doc=feat_builder,
                                             parsed_col='description')
                .build())

# generate scatterplot and save to HTML file
# set use_non_text_features=True to use labeled Empath topics/categories
# set topic_model_term_lists=feat_builder.get_top_model_term_lists to ensure relevant terms are bolded in excerpts
html = st.produce_scattertext_explorer(empath_corpus,
                                       category='4.0-5.0',
                                       category_name='Star Rating 4.0-5.0',
                                       not_category_name='All Other Ratings',
                                       width_in_pixels=1000,
                                       use_non_text_features=True,
                                       use_full_doc=True,
                                       topic_model_term_lists=feat_builder.get_top_model_term_lists())
open('term-Empath.html', 'wb').write(html.encode('utf-8'))
```
<p align="center">
  <img width="1300" src="https://github.com/stevhliu/airbnbListingDescriptions/blob/master/imgs/term-Empath.png" />
</p>

#### **Word similarity**

It may also be interesting to create a plot of word similarities to visualize how listing descriptions with high star ratings differ in words used. A word2vec model was used to generate word embeddings of the listing descriptions, and then tSNE (t-distributed stochastic neighbor embedding) was used to reduce dimensionality.

A word2vec model is simply a vector representation of a word, and maps words to a vector of real numbers (otherwise known as a word embedding). Two common word embedding models are the continuous bag of words model and the skip-gram model. The latter assumes that context words are generated from a central target word, while the former assumes the opposite.

Generating word embeddings can create highly dimensional data, which is why its useful to employ a dimensionality reduction technique like tSNE. tSNE is an implementation of manifold learning, a technique for learning non-linear structure in data. The advantages of tSNE include reducing tendency of crowding points around the center, and exposing natural clusters within the data. Read here for more details about tSNE.

```python
# import necessary libraries
from sklearn.manifold import TSNE
from gensim.models.word2vec import Word2Vec

# split sentences in listing description with st.whitespace_nlp_with_sentences
listing['parse'] = listing['description'].apply(st.whitespace_nlp_with_sentences)

# create a stop-listed corpus of unigram terms
corpus = (st.CorpusFromParsedDocuments(listing, 
                                       category_col='review_scores_rating', 
                                       parsed_col='parse')
          .build()
          .get_stoplisted_unigram_corpus())

# generate scatterplot and save to HTML file
# learn word embeddings with a word2vec model (specify more worker threads for faster processing)
# use t-SNE projection model to visualize word similarity in two-dimensions
html = st.produce_projection_explorer(corpus,
                                      word2vec_model=Word2Vec(workers=4),
                                      projection_model=TSNE(),
                                      category='4.0-5.0',
                                      category_name='Star Rating 4.0-5.0',
                                      not_category_name='All Other Ratings')  
open('wordSimilarity.html', 'wb').write(html.encode('utf-8'))
```
<p align="center">
  <img width="1000" src="https://github.com/stevhliu/airbnbListingDescriptions/blob/master/imgs/tSNE.png" />
</p>

BERT
------
Bidirectional Encoder Representations from Transformers (BERT) is a state-of-the-art model released by Google in late 2018. It is a method for adapting pre-trained language representations for downstream NLP tasks. What makes BERT truly unique is its bidirectionality, meaning, it captures contextual information to the left and right of a word. Training bidirectional models are inherently difficult though because it would allow the word that's being predicted to indirectly see itself in a multi-layered context. This hurdle is overcome by randomly masking some words, and then asking BERT to predict the masked word. To train BERT to understand relationships between two sentences, during the pre-training process, BERT is further tasked with predicting whether one sentence follows another.

There are two stages in using BERT: pre-training and fine-tuning.

Pre-training is the process of training the BERT model on an extremely large text corpus (Wikipedia) to learn language representations. This is a highly expensive task (four days of training on 4-16 cloud TPUs!), but thankfully, the researchers at Google have already completed this step and have publicly released their pre-trained models. All we have to do is download these pre-trained models. For this project, we will use the Uncased BERT-Base model, which simply means all the text has been lowercased, and accent markers have been stripped. The BERT-Base model architecture contains 12 Transformer blocks, 768 hidden units, and 12 attention heads.

Fine-tuning is the stage where we adapt the BERT model to our classification task. On a CPU, this process took 14 hrs, so we recommend using Google's Colaboratory notebook, and accelerate the process by taking advantage of their cloud GPUs.

The following code is adapted from Google's BERT FineTuning with Cloud TPU notebook.

WordPiece tokenization
-----
It is common in NLP to tokenize text such that each word is a separate entity. BERT uses WordPiece model tokenization, where each word is further segmented into sub-word units. A special [CLS] (classifier) token is appended to the beginning of the text, and a [SEP] (separator) token is inserted at the end. The WordPiece tokens are generated by optimizing the number of tokens, such that the resulting corpus contains the minimal number of WordPieces when segmented according to the model. By creating sub-word units, WordPiece allows BERT to capture out-of-vocabulary words and store only 30,522 words, which effectively gives it an evolving word dictionary.

In summary, WordPiece tokenization achieves a good balance between vocabulary size and rare words.

```python
# create a tokenizer from uncased BERT model
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'

def create_tokenizer_from_hub():
    with tf.Graph().as_default():
        bert_module=hub.Module(BERT_MODEL_HUB)
        tokenization_info=bert_module(signature='tokenization_info', as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info['vocab_file'],
                                                  tokenization_info['do_lower_case']])
    return bert.tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub()
```

Predict
-----
```python
# create function to take new predictions
def getPrediction(text):
    labels = ['no_review', '0.0-1.0', '1.0-2.0', '2.0-3.0', '3.0-4.0', '4.0-5.0']
    input_examples = [run_classifier.InputExample(guid=None, text_a = x, text_b = None, label = 0) for x in text]
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(text, predictions)]
    
# insert text from listing description
text = ['''We live in an 1895 Victorian top flat with 12 foot ceilings on the best block in San Francisco. 
           Our calendar is always up to date. Location Location Location~ Our home is close to the Mission, 
           Lower Haight, and 2.5 blocks from the Castro theater. The underground is 3 blocks away, with a 
           street car even closer. The room is small and cosy, but a great price for one of the most 
           expensive neighborhoods in the US!''',
    '']
  
predictions = getPrediction(text)
predictions[0]
```
