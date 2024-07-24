import pandas as pd
import tiktoken
import os
# import openai

# from openai.embeddings_utils import get_embedding

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def item():
    a_df = pd.read_csv('archive/anime.csv', usecols=['Aired'])

    anime = pd.read_csv('archive/anime_with_synopsis.csv')
    # print(anime.head())
    anime['Aired'] = a_df['Aired']
    anime = anime.dropna()

    anime['combined_info'] = anime.apply(lambda row: f"Title: {row['Name']}. Overview: {row['sypnopsis']} Genres: {row['Genres']} Aired: {row['Aired']}", axis=1)
    # print(anime['combined_info'][0])

    anime[['combined_info']].to_csv('archive/anime_updated.csv', index=False)
    pd.read_csv('archive/anime_updated.csv')

    loader = CSVLoader(file_path="archive/anime_updated.csv")
    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = Chroma.from_documents(texts, embeddings)

    # query = "I'm looking for an animated action movie. What could you suggest to me?"
    # docs = docsearch.similarity_search(query, k=1)
    # print(docs)
    # llm = Ollama(model="llama3:70b")

    # qa = RetrievalQA.from_chain_type(llm,
    #                                 chain_type="stuff", 
    #                                 retriever=docsearch.as_retriever(), 
    #                                 return_source_documents=True)

    # query = "I'm looking for an action anime. What could you suggest to me?"
    # result = qa({"query": query})
    # print(result['result'])
    # print("*"*200)
    # print(result['source_documents'][0])

    return docsearch

def user(docsearch):
    template_prefix = """    You are a movie recommender system that help users to find anime that match their preferences. 
    Use the following pieces of database to answer the question at the end. 
    For each question, take into account the database and the personal information provided by the user.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    """

    user_info = """    This is what we know about the user, and you can use this information to better tune your research:
    Age: {age}
    Gender: {gender}
    Genre: {genre}
    Aired: {aired}
    """

    template_suffix= """    Question: {question}
    Your response:
    """

    user_info = user_info.format(age = 29, gender = 'male', genre = 'Shounen', aired = 'latest')

    COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
    print(COMBINED_PROMPT)

    PROMPT = PromptTemplate(
        template=COMBINED_PROMPT, input_variables=["context", "question"])

    llm = Ollama(model="llama3:70b")

    qa = RetrievalQA.from_chain_type(llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(),
        return_source_documents=True, 
        chain_type_kwargs={"prompt": PROMPT})

    query = "Can you recommend some animes that suit my taste?"
    result = qa({'query':query})

    print(result['result'])
    print("*"*200)
    print(result['source_documents'])

if __name__ == "__main__":
    docsearch = item()
    user(docsearch)