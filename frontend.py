import streamlit as st
import PyPDF2
from io import BytesIO
import os
from pdfminer.high_level import extract_text


from GPTVectorStoreIndex import simple_llama    

# Title of the web app
st.title('Chat AWG')

# Create a file uploader widget
file_uploader = st.file_uploader("First, Upload Your Files as PDFs")

file_names = []
if file_uploader:
  file_names.append(file_uploader.name)

# If any files are uploaded, write them to the "data" folder
for file_name in file_names:
  file_path = os.path.join('data', file_name) 
  if os.path.exists(file_path):
    # st.warning("File name already exists.")
    pass
  else: 
    with open(file_path, "wb") as f:
      f.write(file_uploader.read())
    # If the file is a PDF, convert it to a TXT file
    if file_name.endswith(".pdf"):
      text = extract_text(f"data/{file_name}")
      with open(f"data/{file_name[:-4]}.txt", "w") as f:
        f.write(text)
    st.success("PDF saved successfully!")

  # Create a search bar
  query = st.text_input("Next, enter a query about your data: ")

  # If the user enters a query, search through the uploaded documents
  if query != "":
    response = simple_llama(query)
    st.write(response)




