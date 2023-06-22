from dotenv import load_dotenv

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage


print(f"done with imports")

load_dotenv()  # take environment variables from .env.

# sequentially go through each retrieved Node. Use a Question-Answer Prompt for the first Node, 
# and use a Refine Prompt for subsequent Nodes. Make a separate LLM call per Node.

def simple_llama(query):
    try:
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        index = load_index_from_storage(storage_context)
    except:
        directory_path = 'data'
        documents = SimpleDirectoryReader(directory_path).load_data()
        index = GPTVectorStoreIndex.from_documents(documents)

        index.storage_context.persist()


    # llama by default uses text-davinci-003
    query_engine = index.as_query_engine()

    return query_engine.query(query)
