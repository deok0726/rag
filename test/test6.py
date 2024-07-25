import langchain
from langchain import hub
from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

langchain.debug = True
# Loads the latest version
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
# prompt = """    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

# Question: {question} 

# Context: {context} 

# Answer:

# """

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
loader = CSVLoader(file_path="../archive/anime_updated.csv")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# PROMPT = PromptTemplate(
#         template=prompt, input_variables=["question", "context"])
llm = Ollama(model="llama3:70b")
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": prompt}
)

question = "Can you recommend some animes that suit my taste?"
result = qa_chain({"query": question})
print(result["result"])