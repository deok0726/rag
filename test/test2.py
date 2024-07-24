
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from dotenv import load_dotenv
import os

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone, PodSpec
# from langchain_pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_pinecone import Pinecone as PCS

load_dotenv()
# os.environ["PINECONE_API_KEY"]

# Note: If you're using PyPDFLoader then it will split by page for you already

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
index_name = 'clova'

# pc.delete_index(index_name)
# print(f"The index '{index_name}' is removed.")
# pc.create_index(
#     name=index_name,
#     dimension=384,
#     metric="cosine",
#     spec=PodSpec(
#         environment="gcp-starter"
#     )
# )
# print(f"Index '{index_name}' created successfully.")

# # Add additional data
# loader = TextLoader("state_of_the_union_small.txt", encoding='utf-8')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# PineconeStore.from_documents(docs, index_name=index_name, embedding=embeddings)

# loader = UnstructuredPDFLoader("llm_gpt.pdf")
# data = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(data)

# PineconeStore.from_documents(docs, embeddings, index_name=index_name)

# query = "how can I fine-tune LLM model?"
query = "반도체 산업 해외 시장 상황이 어때?"
pcs = PCS(index=index_name, embedding=embeddings)
docs = pcs.similarity_search(query, k=3)
print(docs[0].page_content)