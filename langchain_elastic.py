from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from elasticsearch import Elasticsearch
from langchain.chains.question_answering import load_qa_chain

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Ollama

from dotenv import load_dotenv
import sys
import os

load_dotenv()

# os.environ['HUGGINGFACEHUB_API_TOKEN']

def insert(index, type):
    if type == 'text':
        loader = TextLoader("state_of_the_union.txt", encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
    elif type == 'pdf':
        loader = UnstructuredPDFLoader("materials.pdf")
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = ElasticsearchStore.from_documents(
        docs,
        embedding,
        es_url="http://localhost:9200",
        index_name=index,
    )

    db.client.indices.refresh(index=index)


def retrieve(index, strategy):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if strategy == 'approx':
        db = ElasticsearchStore(
            embedding=embedding,
            es_url="http://localhost:9200",
            index_name=index,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
        )

        results = db.similarity_search(
            query="What did the president say about Ketanji Brown Jackson?", k=4
        )
        # print(results[0])
        for t in results:
            print(t.page_content)
            print("-"*200)

    elif strategy == 'hybrid':
        db = ElasticsearchStore(
            embedding=embedding, 
            es_url="http://localhost:9200", 
            index_name=index,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                hybrid=True,
            )
        )

        results = db.similarity_search(
            "What did the president say about Ketanji Brown Jackson", k=4
        )
        # print(results[0])
        for t in results:
            print(t.page_content)
            print("-"*200)

    # elif strategy == 'elser':
    #     db = ElasticsearchStore(
    #         embedding=embedding, 
    #         es_url="http://localhost:9200", 
    #         index_name=index,
    #         strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(),
    #     )
    #     results = db.similarity_search(
    #         "What did the president say about Ketanji Brown Jackson", k=4
    #     )
    #     print(results[0])

    elif strategy == 'exact':        
        db = ElasticsearchStore(
            embedding=embedding,
            es_url="http://localhost:9200", 
            index_name=index,
            strategy=ElasticsearchStore.ExactRetrievalStrategy()
        )
        results = db.similarity_search(
            "What did the president say about Ketanji Brown Jackson", k=4
        )
        # print(results[0])
        for t in results:
            print(t.page_content)
            print("-"*200)

    else:
        db = ElasticsearchStore(
            embedding=embedding,
            es_url="http://localhost:9200", 
            index_name=index,
            distance_strategy=strategy
            # distance_strategy="COSINE"
            # distance_strategy="EUCLIDEAN_DISTANCE"
            # distance_strategy="DOT_PRODUCT"
        )
        query = "What did the president say about Ketanji Brown Jackson"
        results = db.similarity_search(query)
        # print(results[0])
        
        for t in results:
            print(t.page_content)
            print("-"*200)


def delete(index):
    es = Elasticsearch(hosts=["http://localhost:9200"])  # Initialize Elasticsearch client
    es.indices.delete(index=index, ignore=[400, 404])


def llm(index, strategy):
    # repo_id = 'abacusai/Smaug-72B-v0.1'
    # repo_id = 'mistralai/Mistral-7B-v0.1'
    # repo_id = 'meta-llama/Llama-2-7b'
    # repo_id = 'microsoft/phi-2'
    
    # llm = HuggingFaceHub(
    #     repo_id=repo_id, 
    #     model_kwargs={"temperature": 0.2, 
    #                 "max_length": 128}
    # )

    # llm = Ollama(model="phi")
    llm = Ollama(model="phi3")
    # llm = Ollama(model="llama3")

    prompt = PromptTemplate(input_variables=["context", "question"],
                            template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
                            {context}
                            Question: {question}
                            Helpful Answer:""")

    
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = ElasticsearchStore(
            embedding=embedding,
            es_url="http://localhost:9200", 
            index_name=index,
            distance_strategy=strategy
            # distance_strategy="COSINE"
            # distance_strategy="EUCLIDEAN_DISTANCE"
            # distance_strategy="DOT_PRODUCT"
        )
    
    # query = "What did the president say about Ketanji Brown Jackson"
    # query = "How can I use Generative AI in material sciences?"
    query = "List up use cases of generative ai in material sciences?"
    docs = db.similarity_search(query)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run(context=docs, question=query)
    # result = llm_chain.invoke(input_documents=docs, question=query)
    print(llm_chain.prompt.format_prompt(context=docs, question=query).to_string())
    print("-"*200)
    print(result)


def chat(index, strategy):
    # repo_id = 'meta-llama/Llama-2-7b'
    # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 128})
    llm = Ollama(model="phi")


    prompt = PromptTemplate(input_variables=["context", "question"],
                            template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
                            {context}
                            Question: {question}
                            Helpful Answer:"""
                            )
    
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    # chain = load_qa_chain(llm, chain_type="stuff")

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = ElasticsearchStore(
            embedding=embedding,
            es_url="http://localhost:9200", 
            index_name=index,
            distance_strategy=strategy
            # distance_strategy="COSINE"
            # distance_strategy="EUCLIDEAN_DISTANCE"
            # distance_strategy="DOT_PRODUCT"
        )
    
    # query = "What did the president say about Ketanji Brown Jackson"
    query = "How many alliances does United States have?"
    docs = db.similarity_search(query)

    result = chain.run(input_documents=docs, question=query)
    print(LLMChain(prompt=prompt, llm=llm).prompt.format_prompt(context=docs, question=query).to_string())
    print("-"*200)
    print(result)

if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd[0] == 'i':
        insert(cmd[1], cmd[2])
    elif cmd[0] == 'r':
        # print(cmd[1], cmd[2])
        retrieve(cmd[1], cmd[2])
    elif cmd[0] == 'd':
        delete(cmd[1])
    elif cmd[0] == 'l':
        llm(cmd[1], cmd[2])
    elif cmd[0] == 'c':
        chat(cmd[1], cmd[2])