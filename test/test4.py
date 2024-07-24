from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from elasticsearch import Elasticsearch
from langchain.chains.question_answering import load_qa_chain

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="phi")


# prompt = PromptTemplate(input_variables=["context", "question"],
#                         template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
#                         Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
#                         {context}
#                         Question: {question}
#                         Helpful Answer:"""
#                         )

# chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
chain = load_qa_chain(llm, chain_type="stuff")

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = ElasticsearchStore(
        embedding=embedding,
        es_url="http://localhost:9200", 
        index_name='langchain',
        distance_strategy='COSINE'
        # distance_strategy="COSINE"
        # distance_strategy="EUCLIDEAN_DISTANCE"
        # distance_strategy="DOT_PRODUCT"
    )

query = "What did the president say about Ketanji Brown Jackson"
# query = "How many alliances does United States have?"
docs = db.similarity_search(query)

result = chain.run(input_documents=docs, question=query)
# print(LLMChain(prompt=prompt, llm=llm).prompt.format_prompt(context=docs, question=query).to_string())
print("-"*200)
print(result)