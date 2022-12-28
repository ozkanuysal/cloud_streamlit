# Import important libraries
import pandas as pd
from serpapi import GoogleSearch
import model_pipeline as mp
import urllib.request
import os
import glob

# Import libraries for text preprocessing
import spacy
#import en_core_web_sm
import pandas as pd
#import seaborn as sns
#import numpy as np
import re
#import random
#import nltk

# Intitialize variables 
titles = []
images = []
# Set serp_api_key - https://serpapi.com/ sign up for free plan
GoogleSearch.SERP_API_KEY = "be13155bcdf26b0b3f4735bdb30945f09b7969dd8ae6f71e2b82f72a4e70ecf7"

def data_scrapping(user_product_title, user_product_description, user_product_line, user_product_id):
    params = {
        #"q": user_product_title+user_product_line+user_product_description+user_product_id+ "buy",
        "q": user_product_line+user_product_title+user_product_description+user_product_id+ "buy",
        #"engine": "google",
        #"location": "India, Austin, Texas, United States",
        "location": "India", 
        #"device": "desktop|mobile|tablet",
        #"hl": "Google UI Language",
        "h1": "en",
        #"gl": "Google Country",
        "gl": "IN",
        #"safe": "Safe Search Flag",
        "start": "0",
        "num": 500,
        #"num": 1,
        #"start": "Pagination Offset",
        #"api_key": "Your SERP API Key", 
        # To be match
        #"tbm": "nws|isch|shop", # nws-news, shop-shop, isch-images
        "tbm": "shop", 
        # To be search
        #"tbs": "custom to be search criteria",
        "tbs": "il:cl",
        #"tbs": "vw:l,mr:1,price:1,ppr_min:1000,ppr_max:2000,color:specific,color_val:black,avg_rating:400",
        #"tbs": "vw:l,mr:1,price:1,ppr_min:1000,ppr_max:2000,color:specific,color_val:black,pdtr0:871889%7C872292",
        # allow async request
        #"async": "true|false",
        # output format
        "output": "json"
    }

    # define the search search
    search = GoogleSearch(params)
    # override an existing parameter
    # search.params_dict["location"] = "India"
    # search format return as raw html
    # html_results = search.get_html()
    # parse results
    #  as python Dictionary
    # dict_results = search.get_dict()
    #  as JSON using json package
    json_results = search.get_json()
    #  as dynamic Python object
    # object_result = search.get_object()
    return json_results

# Function to save product image in local drive
def save_image(product_position, product_id, product_title, product_url) :
    print("product_title:", product_title)
    print("product_id:", product_id)
    # Using pathlib, specify where the image is to be saved
    cwd = os.getcwd()
    #print("CWD:", cwd)
    downloads_path = str(cwd+"/images")
    # create new directory if not exists
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)
    else:
        if product_position == 1:
            files = glob.glob(downloads_path+'/*')
            for f in files:
                os.remove(f)    
                titles.clear()
                images.clear()     
    # Form a full image path by joining the path to the 
    picture_path  = os.path.join(downloads_path, product_id+".png")
    # Using "urlretrieve()" from urllib.request save the image 
    urllib.request.urlretrieve(product_url, picture_path)
    titles.append(product_title)
    images.append(downloads_path+"/"+product_id+".png")
    print("image saved")

# Function to set data to model_pipeline
def set_data(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path):
    mp.image_title_dataset(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, titles, images)   

# Function to get products via serpAPI
def get_products(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path):
    json_results = data_scrapping(user_product_title, user_product_description, user_product_line, user_product_id)
    results = []

    for shopping_results in json_results['shopping_results']:
        results.append({
                    #'product_id': shopping_results['product_id'],
                    'title': shopping_results['title'],
                    #'product_link': shopping_results['product_link'],
                    #'source': shopping_results['source'],
                    #'price': shopping_results['price'],
                    #'extracted_price': shopping_results['extracted_price'],
                    'price':int(shopping_results['extracted_price']),
                    #'shipping': inline_shopping_results['shipping'],
                    #'rating': shopping_results['rating'],
                    #'reviews': shopping_results['reviews'],
                    'image': shopping_results['thumbnail'],
                })
        #Declare variables to get product details 
        product_position = shopping_results['position']
        product_id = shopping_results['product_id']
        print("origional title:", shopping_results['title'])
        product_title = preprocessing(shopping_results['title'])
        product_url = shopping_results['thumbnail']
        # Call save_image function to save product_image
        save_image(product_position, product_id, product_title, product_url)
    #print(images)
    #print(captions)
    # user_input - text preprocessing
    user_product_title = preprocessing(user_product_title)
    user_product_description = preprocessing(user_product_description)
    user_product_line = preprocessing(user_product_line)
    set_data(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path)
    products_data = pd.DataFrame(results)
    return products_data


# Define function for data preprocessing
def preprocessing(sentence):
  # Load nlp
  nlp = spacy.load('en_core_web_sm')
  # Text normalization
  sentence = sentence.lower() # Lower text
  # Remove punctuation via regular expression
  sentence = re.sub(r"@[A-Za-z0-9]+", ' ', sentence)
  sentence = re.sub(r"https?://[A-Za-z0-9./]+", ' ', sentence)
  sentence = sentence.replace('.', '')
  # Tokenization - split doc into paragraphs, sentences, words, characters etc.
  # tokenization in sentences
  tokens = []
  # 'nlp' object is used to create documents into linguistic annotations
  #tokens = [token.text for token in nlp(sentence) if not (token.is_stop or token.like_num or token.is_punct or token.is_space or len(token) == 1)]
  tokens = [token.text for token in nlp(sentence) if not (token.is_stop or token.is_punct or token.is_space or len(token) == 1)]
  tokens = ' '.join([element for element in tokens])
  print("token:", tokens)  
  return tokens
