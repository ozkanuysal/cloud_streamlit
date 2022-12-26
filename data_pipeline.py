# Import important libraries
import pandas as pd
from serpapi import GoogleSearch
import model_pipeline as mp
import urllib.request
import os
import glob

# Import libraries for text preprocessing
import spacy
import pandas as pd
import re

# Intitialize variables 
titles = []
images = []
# Set serp_api_key - https://serpapi.com/ sign up for free plan
GoogleSearch.SERP_API_KEY = "1d6a859b3e556147335cf2043449d66723d746ff13382b669937572b0c804aa6"

def data_scrapping(user_product_title, user_product_description, user_product_line, user_product_id):
    params = {
        "q": user_product_line+user_product_title+user_product_description+user_product_id+ "buy",
        "location": "New York", 
        "h1": "en",
        "gl": "US",
        "start": "0",
        "num": 500,
        "tbm": "shop", 
        "tbs": "il:cl",
        "output": "json"
    }

    # define the search search
    search = GoogleSearch(params)
    json_results = search.get_json()
    return json_results

# Function to save product image in local drive
def save_image(product_position, product_id, product_title, product_url) :
    cwd = os.getcwd()
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
    urllib.request.urlretrieve(product_url, picture_path)
    titles.append(product_title)
    images.append(downloads_path+"/"+product_id+".png")

# Function to set data to model_pipeline
def set_data(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path):
    mp.image_title_dataset(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, titles, images)   

# Function to get products via serpAPI
def get_products(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path):
    json_results = data_scrapping(user_product_title, user_product_description, user_product_line, user_product_id)
    results = []
    print(json_results)

    for shopping_results in json_results['shopping_results']:
        results.append({
                    'product_id': shopping_results['product_id'],
                    'title': shopping_results['title'],
                    'product_link': shopping_results['product_link'],
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
  tokens = []
  tokens = [token.text for token in nlp(sentence) if not (token.is_stop or token.is_punct or token.is_space or len(token) == 1)]
  tokens = ' '.join([element for element in tokens])
  print("token:", tokens)  
  return tokens
