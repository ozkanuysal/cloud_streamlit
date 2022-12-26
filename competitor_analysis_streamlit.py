# Import libraries
import streamlit as st
import data_pipeline as dp
import model_pipeline as mp
from PIL import Image
import numpy as np 
import streamlit as st 
import pandas as pd
import os 
import glob

st.set_page_config(
    layout="wide"
)



st.markdown("<h1 style='text-align: center;'>Competitor Analysis</h1>", unsafe_allow_html=True)

# Converting links to html tags
def path_to_image_html(path):
    return '<img src="' + path + '" width="200" >'

col1, col2, col3, col4 = st.beta_columns((1.8, 2, 2, 2))

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

with st.form("my_form"):
    st.markdown("### Product Details")
    col1, col2, col3, col4 , col5= st.beta_columns(5)
    with col1:
            product_line = st.selectbox("Product Line: ",
                     ['Air-Force', 'Air-max', 'Closing', 'Jordan', 'Nike', 'Tablet'])
            with col2:
                product_id = st.text_input("Product Id")
                with col3:
                    product_title = st.text_input("Title (required)")
                    with col4:
                        product_description = st.text_area("Description")
                        with col5:
                            uploaded_file = st.file_uploader(label="Product image", type=['jpg', 'png', 'jpeg'])
                            if uploaded_file is not None:
                                file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
                                user_product_image = load_image(uploaded_file)
                                st.image(user_product_image, width=150)
                                #st.write(file_details)
                                user_product_image_path = save_uploadedfile(uploaded_file)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Search competitors")
    if submitted:
        if not product_title:
            st.error("title required")
        else:
            df = dp.get_products(product_title, product_description, product_line, product_id, user_product_image_path) 

            df1 = pd.DataFrame(columns = ['Product_ID', 'Product_Name','Product Link', 'Price($)', 'Score', 'image'])      

            for i in df.index:
                for j in mp.similar_products_df1.index:
                    if i ==  mp.similar_products_df1['product_index'][j]: 
                        product_id = df['product_id'][i]   
                        product_name = df['title'][i] 
                        product_price = df['price'][i]
                        product_link = df['product_link'][i]
                        product_index = mp.similar_products_df1['product_index'][j]
                        CLIP_similarity_score = mp.similar_products_df1['CLIP_similarity_score'][j]
                        df1=df1.append({'Product_ID':product_id,'Product_Name':product_name,'Product Link':product_link,'Price($)':product_price,'Score':CLIP_similarity_score,'image':df['image'][i]},ignore_index=True)
            df1.reset_index(drop=True)
            df1.sort_values(by='Score', ascending=False)

            st.markdown(
                 df1.sort_values(by='Score', ascending=False).to_html(escape=False,col_space='100px', formatters=dict(image=path_to_image_html)),
                  unsafe_allow_html=True,)
            df1.to_html("webpage.html", escape=False, formatters=dict(image=path_to_image_html))

st.write("Outside the form")

