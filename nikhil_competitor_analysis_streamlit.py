# Import libraries
!pip install serpapi
import streamlit as st
import data_pipeline as dp
import model_pipeline as mp
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
    
