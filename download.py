import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.utilities import ApifyWrapper
from langchain.document_loaders.base import Document

# Load environment variables from a .env file
load_dotenv()

if __name__ == '__main__':
    apify = ApifyWrapper()
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": os.environ.get('WEBSITE_URL')}]},
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator='\n')
    docs = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()

    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
