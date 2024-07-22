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
import sys

load_dotenv()
# os.environ["PINECONE_API_KEY"]

loader = UnstructuredPDFLoader("llm_gpt.pdf")
# loader = PyPDFLoader("../data/field-guide-to-data-science.pdf")
# loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

data = loader.load()

# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your sample document')
# print (f'Here is a sample: {data[0].page_content[:200]}')
sys.exit()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
# print(docs[0])
# print(docs[1])
# print(docs[-1])

# Let's see how many small chunks we have
print (f'Now you have {len(docs)} documents')

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
indexes = pc.list_indexes()
index_name = 'langchain'
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Check if your desired index name is in the list
if index_name in indexes[0]['name']:
    print(f"The index '{index_name}' exists.")
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
else:
    print(f"The index '{index_name}' does not exist.")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=PodSpec(
            environment="gcp-starter"
        )
    )
    print(f"Index '{index_name}' created successfully.")

    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer("jingyeom/korean_embedding_model")
    # docsearch = pc.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

docsearch = PineconeStore.from_documents(docs, embeddings, index_name=index_name)
print(f'{index_name} upserted with documents')

query = "how can I fine-tune LLM model?"
docs = docsearch.similarity_search(query, k=3)

print(docs[0].page_content)

# Add additional data
loader = TextLoader("state_of_the_union_small.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

PineconeStore.from_documents(docs, index_name=index_name, embedding=embeddings)

# Search Only
query = "how can I fine-tune LLM model?"
pcs = PCS(index=index_name, embedding=embeddings)
docs = pcs.similarity_search(query, k=3)
print(docs[0].page_content)