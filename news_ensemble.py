import os, sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA

dotenv_path = os.path.join(os.path.dirname(__file__), 'config/.env')
load_dotenv(dotenv_path)
api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ensemble(path, extract, k):
    docs = []
    if os.path.isdir(path):
        for file in os.listdir(path):
            f = os.path.join(path, file)
            loader = TextLoader(f)
            # print(f"Loaded Text: {f.split('/')[-1]}")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs.extend(text_splitter.split_documents(data))
    else:
            loader = TextLoader(path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs.extend(text_splitter.split_documents(data))
        
    # model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    embeddings = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

    vectorstore_faiss = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore_faiss.as_retriever(search_kwargs={'k': k})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    
    # bm25_results = bm25_retriever.invoke(query)
    # faiss_results = faiss_retriever.invoke(query)
    # bm25_results = bm25_retriever.get_relevant_documents(query)
    # faiss_results = faiss_retriever.get_relevant_documents(query)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    results = ensemble_retriever.get_relevant_documents(extract)
    for i in results:
        print(i.metadata)
        print(i.page_content)
        print("-"*100)

    llm = Ollama(model="llama3.1:70b", temperature=0)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # template = '''
    #             You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    #             Question: {question} 
    #             Context: {context} 
    #             Answer:
    #             '''
    template = '''
                According to the instructions, extract the following insights from the extracted context. Specify in the insight if a specific company name is mentioned. Write the category insight name in Korean only without using parentheses. Use bullets to summarize. Do not use tables. Accuracy is very important.
                Context: {context} 
                Instructions: {instructions} 
                Insights:
                '''
    prompt = PromptTemplate.from_template(template)

    context = format_docs(ensemble_retriever.get_relevant_documents(extract))

    # Print the context for debugging
    print("=== Context ===")
    print(context)
    print("================")
    
    # retrieval_qa_chain = RetrievalQA(
    #         retriever=ensemble_retriever,
    #         llm=llm,
    #         prompt=prompt,
    #         output_parser=StrOutputParser()
    #     )

    rag_chain = (
        {"context": ensemble_retriever | format_docs, "instructions": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    instructions = input('''Summarize the content that has '신용카드', '카드론', '카드대출', '카드페이먼트' for '키워드'. Be sure to include all keywords. If there are no articles with keyword stated above, do not make bullet for the keyword. Group the news articles into groups based on '키워드', using bullet points. Summarize the content of each keyword by 3~5 bullets by reading '제목', and '요약'. Include in the summary if a specific company name is mentioned but do not emphasize with parentheses. Include in the summary if numbers are mentioned. Accuracy is very important. Do not number the groups. Please answer in Korean.''')
    response = rag_chain.invoke(instructions)

    print(f"문서의 수: {len(docs)}")
    print("===" * 20)
    print(f"[HUMAN]\n{instructions}\n")
    print(f"[AI]\n{response}")

if __name__ == "__main__":
    path = '/svc/project/genaipilot/rag/data/news/'
    extract = "'신용카드', '카드론', '카드대출', '카드페이먼트' 등을 키워드로 가지는 카드사 동향 뉴스"
    k = 3
    ensemble(path, extract, k)