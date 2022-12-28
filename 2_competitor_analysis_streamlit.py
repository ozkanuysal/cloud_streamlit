# Import libraries
import streamlit as st
#import data_pipeline as dp
#import model_pipeline as mp
from PIL import Image
import numpy as np 
import streamlit as st 
import os 
import glob

st.set_page_config(
    layout="wide"
)

st.title('Competitor Analysis')

col1, col2, col3, col4 = st.columns((1.8, 2, 2, 2))

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def save_uploadedfile(uploadedfile):
    cwd = os.getcwd()
    #print("CWD:", cwd)
    downloads_path = str(cwd+"/user_image")
    
    # create new directory if not exists
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)           
    
    # Check is directory is empty or not
    if len(downloads_path) != 0:
        files = glob.glob(downloads_path+'/*')
        print("files: ", file_details)
        for f in files:
            os.remove(f)
            
    picture_path  = os.path.join(downloads_path, uploaded_file.name)
    with open(picture_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    
    return  picture_path 
    #st.write(user_product_image_path)
    #user_product_image = load_image(picture_path)
    #st.image(user_product_image, width=200)
    
# Import important libraries
import pandas as pd
#from serpapi import GoogleSearch
#import model_pipeline as mp
import urllib.request
import os
import glob

# Import libraries for text preprocessing
#import spacy
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
#GoogleSearch.SERP_API_KEY = "be13155bcdf26b0b3f4735bdb30945f09b7969dd8ae6f71e2b82f72a4e70ecf7"

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
  
  
  

import streamlit as st
from PIL import Image
import pandas as pd
#from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
#import torch
#from tqdm.auto import tqdm
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from scipy.spatial.distance import cosine

# Declare vaiables for text-image similarity function
index_lst1 = []
score_lst1 = []
similar_products_df1 = pd.DataFrame()

# Declare vaiables for image-image similarity function
index_lst2 = []
score_lst2 = []
similar_products_df2 = pd.DataFrame()

# if you have CUDA or MPS, set it to the active device like this
#device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

#device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model_id = "openai/clip-vit-base-patch32"

# we initialize a tokenizer, image processor, and the model itself
#tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
#processor = CLIPProcessor.from_pretrained(model_id)
#model = CLIPModel.from_pretrained(model_id).to(device)

#  def image_title_dataset(titles, images, user_product_title, user_product_image):
def image_title_dataset(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, titles, images):
    dataset = pd.DataFrame({'product_title': titles, 'product_image': images})
    print(dataset)
    # CLIP_model(dataset, user_product_title, user_product_image) 
    CLIP_model(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, dataset) 

# Function for text to image similarity
def text_image_similarity(text_emb, image_arr):    
    # Text-Image Search
    # calculating the dot product similarity between our query and the images
    text_emb = text_emb.cpu().detach().numpy()
    # define min max scaler
    # scaler = MinMaxScaler()
    # transform data
    # Calculate dot product
    #scores = scaler.fit_transform(np.dot(text_emb, image_arr.T))
    #scores = np.dot(text_emb, image_arr.T)
    # Calculate Cosine-similarity
    #cosine(image_arr.T, text_emb)
    print("Text-Image Similarity Function")
    #print("Before transform/n")
    scores = np.dot(text_emb, image_arr.T)/(np.linalg.norm(text_emb)*np.linalg.norm(image_arr.T))
    #print("scores shape:", scores.shape)
    #print(scores)

    #print("After transform")
    #scores = scaler.fit_transform(scores)
    scores = np.interp(scores, (scores.min(), scores.max()), (0, 1))
    #print("scores shape:", scores.shape)
    #print("scores: "scores)
    
    top_k = 10
    # get the top k indices for most similar vecs
    idx = np.argsort(-scores[0])[:top_k]
    print("top 10:", idx)
    index_lst1.clear()
    score_lst1.clear()
    
    #index.clear()
    #score.clear()
    # display the results
    for i in idx:
        #print(f"{i}: {scores[0][i]}")
        #st.image(images[i], width=200)
        #print("dataset_row index:", i)
        #print(images[i])
        index_lst1.append(i)
        score_lst1.append(scores[0][i])
    
    similar_products_df1.iloc[0:0]
    similar_products_df1['product_index'] = index_lst1
    similar_products_df1['CLIP_similarity_score'] = score_lst1
    print("similar_products_df1=", similar_products_df1)
    
# Function for text to image similarity
def image_image_similarity(user_image_arr, image_arr):    
    # Image-Image Search
    # calculating the dot product similarity between our query and the images
    # define min max scaler
    # scaler = MinMaxScaler()
    # transform data
    # Calculate dot product
    #scores = scaler.fit_transform(np.dot(user_image_arr, image_arr.T))
    #scores = np.dot(user_image_arr, image_arr.T)
    # Calculate Cosine-similarity
    #cosine(image_arr.T, user_image_arr)
    print("Image-Image Similarity Function")
    #print("Before transform/n")
    scores = np.dot(user_image_arr.T, image_arr.T)/(np.linalg.norm(user_image_arr.T)*np.linalg.norm(image_arr.T))
    #print("scores shape:", scores.shape)
    #print(scores)
    
    #print("After transform")
    #scores = scaler.fit_transform(scores)
    scores = np.interp(scores, (scores.min(), scores.max()), (0, 1))
    #print("scores shape:", scores.shape)    
    #print("scores:", scores)
    
    top_k = 10
    # get the top k indices for most similar vecs
    idx = np.argsort(-scores)[:top_k]
    print("top 10:",idx)
    index_lst2.clear()
    score_lst2.clear()
    
    #index.clear()
    #score.clear()
    # display the results
    for i in idx:
        #print(f"{i}: {scores[i]}")
        #st.image(images[i], width=200)
        #print("dataset_row index:", i)
        #print(images[i])
        index_lst2.append(i)
        score_lst2.append(scores[i])
    
    similar_products_df2.iloc[0:0]
    similar_products_df2['product_index'] = index_lst2
    similar_products_df2['Image_Image_similarity_score'] = score_lst2
    print("similar_products_df2=", similar_products_df2)
    
def CLIP_model(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, dataset):
    user_input = user_product_title+user_product_description+user_product_line,
    print("CLIP Model user input: ", user_input)

    ## Create Text Embedding
    # create transformer-readable tokens
    inputs = tokenizer(user_input, return_tensors="pt")
    # use CLIP to encode tokens into a meaningful embedding
    text_emb = model.get_text_features(**inputs)
    print("embedded text shape:",text_emb.shape)
    print("user_product_image_path:",user_product_image_path)
    # Create an empty array
    user_image_arr = None
    #print("user_image_arr:",user_image_arr)

    # extract the image sample from the dataset
    #images = dataset['product_image']
    #print(images)
    ##np.random.seed(0)
    # select 100 random image index values
    ##sample_idx = np.random.randint(0, len(dataset)+1, 100).tolist()
    # extract the image sample from the dataset
    ##images = [Image.open(dataset['product_image'][i]) for i in sample_idx]
    ##print(images)

    images = [Image.open(dataset['product_image'][i]) for i in dataset.index]
    #print(images)
    
    batch_size = 16
    image_arr = None

    for i in tqdm(range(0, len(images), batch_size)):
        # select batch of images
        batch = images[i:i+batch_size]
        # process and resize
        batch = processor(
            text=None,
            images=batch,
            return_tensors='pt',
            padding=True
        )['pixel_values'].to(device)
        # get image embeddings
        batch_emb = model.get_image_features(pixel_values=batch)
        # convert to numpy array
        batch_emb = batch_emb.squeeze(0)
        batch_emb = batch_emb.cpu().detach().numpy()
        # add to larger array of all image embeddings
        if image_arr is None:
            image_arr = batch_emb
        else:
            image_arr = np.concatenate((image_arr, batch_emb), axis=0)
    print("image_arr shape:", image_arr.shape)
    
    # Normalization
    image_arr = image_arr / np.linalg.norm(image_arr, axis=0)
    image_arr.min(), image_arr.max()
    
    # Call text-image similarity function
    text_image_similarity(text_emb, image_arr)
    
    if user_product_image_path != "":
    ## Create Image Embedding
        image = processor(
            text=None,
            #images=Image.open(dataset['product_image'][0]),
            images=Image.open(user_product_image_path),
            return_tensors='pt'
        )['pixel_values'].to(device)
        print("user_image shape:",image.shape)

        user_image_emb = model.get_image_features(image)
        print("user_image_emb shape:",user_image_emb.shape)
        # convert to numpy array
        user_image_emb = user_image_emb.squeeze(0)
        user_image_emb_arr = user_image_emb.cpu().detach().numpy()
        #print("user_image_emb arr:",user_image_emb_arr)
            
        # Normalization
        user_image_arr = user_image_emb_arr / np.linalg.norm(user_image_emb_arr, axis=0)
        #print("user_image_arr after norm:",user_image_arr)
        user_image_arr.min(), user_image_arr.max()

        # Call image to image similarity function
        image_image_similarity(user_image_arr, image_arr)

        
    
    
    
    
    
    
    
    
    
    

with col1:
    st.markdown("### Product Details")
    product_line = st.text_input("Product Line")
    product_id = st.text_input("Product Id")
    product_title = st.text_input("Title (required)")
    product_description = st.text_area("Description")

    uploaded_file = st.file_uploader(label="Choose a product image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        # To read file as bytes:
        #bytes_data = uploaded_file.getvalue()
        #st.write(bytes_data)
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        user_product_image = load_image(uploaded_file)
        st.image(user_product_image, width=150)
        #st.write(file_details)
        user_product_image_path = save_uploadedfile(uploaded_file)
    # check if the button is pressed or not
    if(st.button('Show competitiors products')):
    
        if not product_title:
            st.error("title required")
        else:
            #st.write("Title: ", product_title)
            #st.write("Description: ", product_description)
            #st.write(dp.data_scrapping(product_title, product_description))
            #st.table(dp.get_products(product_title, product_description))
            # Converting links to html tags
            #st.image(dp.get_products(product_title, product_description))
            df = dp.get_products(product_title, product_description, product_line, product_id, user_product_image_path)
            #st.table(df)
            with col2:
                st.markdown("### Scrapped Products - Google shopping")
                for index in df.index:
                    product_name = df['title'][index]
                    product_price = df['price'][index]
                    st.write(f"Product_No: {index}  \n Product_Name: {product_name}  \n Price: {product_price}")
                    #st.write("**Product_Name:** ", product_name)
                    #st.write("**Price:** ", product_price)
                    st.image(df['image'][index], width=150)
            
            with col3:     
                st.markdown("### Similar Competitor Products")   
                st.table(mp.similar_products_df1)        

                for i in df.index:
                    for j in mp.similar_products_df1.index:
                        if i ==  mp.similar_products_df1['product_index'][j]:   
                            product_name = df['title'][i]
                            product_price = df['price'][i]
                            product_index = mp.similar_products_df1['product_index'][j]
                            CLIP_similarity_score = mp.similar_products_df1['CLIP_similarity_score'][j]
                            #st.write("**Product_Name:** ", product_name)
                            #st.write("**Price:** ", product_price)
                            #st.write("**Product_No:** ", product_index)
                            #st.write("**CLIP_similarity_score:** ", CLIP_similarity_score)
                            st.write(f"Product_No: {product_index}  \n Product_Name: {product_name}  \n Price: {product_price}  \n CLIP_similarity_score: {CLIP_similarity_score}")
                            st.image(df['image'][i], width=150)
            with col4:
                st.markdown("### Similar Competitor Products")   
                st.table(mp.similar_products_df2)        

                for i in df.index:
                    for j in mp.similar_products_df2.index:
                        if i ==  mp.similar_products_df2['product_index'][j]:   
                            product_name = df['title'][i]
                            product_price = df['price'][i]
                            product_index = mp.similar_products_df2['product_index'][j]
                            Image_image_similarity_score = mp.similar_products_df2['Image_Image_similarity_score'][j]
                            #st.write("**Product_Name:** ", product_name)
                            #st.write("**Price:** ", product_price)
                            #st.write("**Product_No:** ", product_index)
                            #st.write("**CLIP_similarity_score:** ", CLIP_similarity_score)
                            st.write(f"Product_No: {product_index}  \n Product_Name: {product_name}  \n Price: {product_price}  \n Image_image_similarity_score: {Image_image_similarity_score}")
                            st.image(df['image'][i], width=150)
    
