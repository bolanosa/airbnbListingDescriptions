#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import colorlover as cl
import numpy as np
from flask import Flask
from flask_cors import CORS
import os

app = Dash(__name__)
server = app.server

layout = html.Div([

    dcc.Markdown('''
    # Analyzing Airbnb listing descriptions
    '''.replace('  ', ''), className='container',
    containerProps={'style': {'maxWidth': '650px'}}),

    html.Img(src='/assets/SanFrancisco.jpg', width='100%'),

    dcc.Markdown('''

    ***

    Airbnb is an online community marketplace for people to book a complete travel experience, including
    accommodations and locally curated activities. One of the first interactions a guest has with a home is
    the listing description. The listing refers to the description of the home and provides an opportunity to
    highlight unique features of your listing. As it turns out, the listing description occupies a decisive
    role in how guests judge the overall quality of a home.

    > This is a companion site for exploring Airbnb listing descriptions. Check out my article for my own
    analysis. In the meantime, scroll down to see how price and ratings are related and textual analysis of
    listing descriptions.

    '''.replace('  ', ''), className='container',
    containerProps={'style': {'maxWidth': '650px'}}),

    dcc.Markdown('''

    ***

    ## Term associations

    Currently, there aren't many good options for text visualization. Word clouds are perhaps the most common
    method for text visualization, but they can be hard to interpret. It can be challenging to compare the
    sizes of two non-horizontally adjacent words, and longer words can appear to have an outsized impact
    simply because they occupy more space.

    To avoid these pitfalls, we use [Scattertext](https://github.com/JasonKessler/scattertext), a visualization
    tool that presents a scatterplot, where each axis corresponds to the frequency of a term and its related
    category. Terms used commonly in all listing descriptions are located in the upper-right corner of the
    plot (home, house, kitchen, apartment, room). It gets more interesting when we look at the upper-left and
    bottom-right corners. These are the terms that are most commonly associated with high star ratings and
    all other ratings respectively. The colors also help identify word association. Those terms that are more
    aligned with high star ratings are blue, and those more associated with all other ratings are red.

    Clicking on a word brings up an excerpt from where it was used in the listing description. There is also a
    search function if you want to look up a specific word. Finally, Scattertext is unable to compare
    multiple groups at once, which is why we need to take a one-to-many visualization approach.

    '''.replace('  ', ''), className='container',
    containerProps={'style': {'maxWidth': '650px'}}),

    html.Iframe(style={'border':'none'},
                srcDoc=open('term-associations.html', 'r').read(),
                width='100%', height='800'),

    dcc.Markdown('''

    ***

    ## Topic categories

    It is helpful to understand what types of topics are being mentioned in the listing descriptions that
    correspond with a high star rating. Hosts can model and incorporate these themes into their own listing
    descriptions, while learning which ones to avoid.

    [Empath](https://hci.stanford.edu/publications/2016/ethan/empath-chi-2016.pdf) allows us to learn the
    topic signals within listing descriptions, by generating and validating categories from a few seed terms.
    For example, the terms vintage, trendy and fashion would fall under the hipster category in Empath.
    Integrating Empath with Scattertext, we can build a scatterplot of topics associated with high star
    ratings and all other ratings.

    '''.replace('  ', ''), className='container',
    containerProps={'style': {'maxWidth': '650px'}}),

    html.Iframe(style={'border':'none'},
                srcDoc=open('term-Empath.html', 'r').read(),
                width='100%', height='800'),

    dcc.Markdown('''

    ***

    ## Word similarities

    It may also be interesting to create a plot of word similarities to visualize how listing descriptions with
    high star ratings differ in terms used. We create a word2vec model to generate word embeddings of the listing
    descriptions, and then use t-SNE (t-distributed stochastic neighbor embedding) to reduce dimensionality and
    project it onto a 2D space.

    '''.replace('  ', ''), className='container',
    containerProps={'style': {'maxWidth': '650px'}}),

    html.Iframe(style={'border':'none'},
                srcDoc=open('wordSimilarity.html', 'r').read(),
                width='100%', height='800')

])


app.layout = layout


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
