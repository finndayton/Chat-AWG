import logging
import sys
import os 
from dotenv import load_dotenv


from langchain import OpenAI
from llama_index import LLMPredictor
from langchain.document_loaders import TextLoader
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
import torch
from llama_index.llm_predictor import HuggingFaceLLMPredictor
from llama_index import StorageContext, load_index_from_storage


print(f"done with imports")

load_dotenv()  # take environment variables from .env.

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# llm_predictor = LLMPredictor(
#     llm=OpenAI(
#         openai_api_key= os.getenv('OPENAI_API_KEY'),
#         temperature=1, 
#         model_name='text-ada-001', 
#         max_tokens=1000
#     ),
# )

# directory_path = '/data/small.txt'

# print(f"commencing load_data on {directory_path}")
# print(f'documents:      {documents}')


# hf_predictor = HuggingFaceLLMPredictor(
#     max_input_size=2048, 
#     max_new_tokens=256,
#     temperature=0.25,
#     do_sample=False,
#     tokenizer_name="Writer/camel-5b-hf",
#     model_name="Writer/camel-5b-hf",
#     device_map="auto",
#     tokenizer_kwargs={"max_length": 2048},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )

# print(f"creating service_context")
# service_context = ServiceContext.from_defaults(chunk_size_limit=512, llm_predictor=hf_predictor)

# print(f"creating index")

def simple_llama(query):
    try:
        storage_context = StorageContext.from_defaults(persist_dir='./storage')

        index = load_index_from_storage(storage_context)
    except:
        directory_path = 'data'
        documents = SimpleDirectoryReader(directory_path).load_data()
        index = GPTVectorStoreIndex.from_documents(documents)

        index.storage_context.persist()

    # rebuild storage context
    # storage_context = StorageContext.from_defaults(persist_dir='./storage')

    # load index
    # index = load_index_from_storage(storage_context)

    # llama by default uses text-davinci-003
    query_engine = index.as_query_engine()

    return query_engine.query(query)

    # while True:
    #     query = input("Enter a Query: ")

    #     if query == "stop":
    #         break
    
    #     response = query_engine.query(query)
    #     print(response)
    #     print()
