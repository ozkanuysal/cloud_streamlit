import streamlit as st
from PIL import Image
import pandas as pd
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine

# Declare vaiables for text-image similarity function
index_lst1 = []
score_lst1 = []
similar_products_df1 = pd.DataFrame()


# if you have CUDA or MPS, set it to the active device like this
#device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model_id = "openai/clip-vit-base-patch32"

# we initialize a tokenizer, image processor, and the model itself
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id, from_tf=True).to(device)

#  def image_title_dataset(titles, images, user_product_title, user_product_image):
def image_title_dataset(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, titles, images):
    dataset = pd.DataFrame({'product_title': titles, 'product_image': images})
    print(dataset)
    # CLIP_model(dataset, user_product_title, user_product_image) 
    CLIP_model(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, dataset) 

# Function for text to image similarity
def txtimg_img_similarity(text_emb, img_emb, image_arr):    
    text_emb = text_emb.cpu().detach().numpy()
    print("Text-Image Similarity Function")
    #print("Before transform/n")
    scores = np.dot(text_emb+img_emb, image_arr.T)/(np.linalg.norm(text_emb)*np.linalg.norm(image_arr.T))

    #scores = scaler.fit_transform(scores)
    scores = np.interp(scores, (scores.min(), scores.max()), (0, 1))
    
    top_k = 10
    # get the top k indices for most similar vecs
    idx = np.argsort(-scores[0])[:top_k]
    print("top 10:", idx)
    index_lst1.clear()
    score_lst1.clear()

    for i in idx:
        index_lst1.append(i)
        score_lst1.append(scores[0][i])
    
    similar_products_df1.iloc[0:0]
    similar_products_df1['product_index'] = index_lst1
    similar_products_df1['CLIP_similarity_score'] = score_lst1
    print("similar_products_df1=", similar_products_df1)
    

    
def CLIP_model(user_product_title, user_product_description, user_product_line, user_product_id, user_product_image_path, dataset):
    user_input = user_product_title+user_product_description+user_product_line,

    ## Create Text Embedding
    # create transformer-readable tokens
    inputs = tokenizer(user_input, return_tensors="pt")
    # use CLIP to encode tokens into a meaningful embedding
    text_emb = model.get_text_features(**inputs)
    print("embedded text shape:",text_emb.shape)
    print("user_product_image_path:",user_product_image_path)
    # Create an empty array
    user_image_arr = None
    images = [Image.open(dataset['product_image'][i]) for i in dataset.index]
    
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

    
    if user_product_image_path != "":
        image = processor(
            text=None,
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

        txtimg_img_similarity(text_emb ,user_image_arr , image_arr)

    
