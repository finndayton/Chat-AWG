import streamlit as st
import PyPDF2
from io import BytesIO
import os

from GPTVectorStoreIndex import simple_llama    

# Title of the web app
st.title('Chat AWG')

# User input
# user_input = st.text_input("Enter some text", 'Type Here')

# Display the input
# st.write('You entered: ', user_input)

# Create a file uploader widget
file_uploader = st.file_uploader("First Upload Your Files as PDFs")

# Get the file names of all uploaded files
file_names = []
if file_uploader:
  file_names.append(file_uploader.name)

# Check if the file names already exist
for file_name in file_names:
  if os.path.exists(file_name):
    st.warning("File name already exists.")
  else:
    # Save the contents of the uploaded PDF to a file
    file_path = os.path.join('data', file_name)
    with open(file_path, "wb") as f:
      f.write(file_uploader.read())

    st.success("PDF saved successfully!")

# Display the file names of all uploaded files
# st.write("File names:", file_names)


# search bar for queries 
query = st.text_input("Enter a Query about your data: ")
response = simple_llama(query)
st.write(response)





