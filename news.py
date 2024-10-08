import time
import os, sys
import ollama
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone, ServerlessSpec

dotenv_path = os.path.join(os.path.dirname(__file__), 'config/.env')
load_dotenv(dotenv_path)
api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def embedding_text(path, index_name):    
    loader = TextLoader(path)
    print(f"Loaded Text: {path}")
    data = loader.load()
    # print (f'You have {len(data)} document(s) in your data')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # print (f'Now you have {len(docs)} documents')

    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    embeddings = SentenceTransformer('jhgan/ko-sroberta-multitask')

    index_names = []
    for index in indexes:
        index_names.append(index['name'])
    print(f"Existing Indexes: {index_names}")

    if index_name in index_names:
        print(f"The index '{index_name}' exists.")
    else:
        print(f"The index '{index_name}' does not exist.")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created successfully.")

    index = pc.Index(index_name)

    vectors_to_upsert = []
    file_name = path.split('/')[-1]
    for idx, doc in enumerate(docs):
        text = doc.page_content
        vector = embeddings.encode(text).tolist()
        # vector_dimension = len(vector)
        # print(vector_dimension)
        time_stamp = int(time.time())
        doc_id = f"doc_{time_stamp}"
        metadata = {'text': text,
                    'source': file_name}
        vectors_to_upsert.append((doc_id, vector, metadata))

    index.upsert(vectors=vectors_to_upsert, namespace='news')

    print(f'{index_name} upserted with documents: {file_name}')
    print(index.describe_index_stats())

def retrieve(question, index_name):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    ollama_model = 'llama3.1:70b'

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    query_vector = model.encode(question).tolist()
    result = index.query(namespace="news", vector=[query_vector], top_k=9, include_metadata=True)
    
    for match in result['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}")
        # print(f"ID: {match['id']}, Score: {match['score']}, Text: {match['metadata']['text']}")
    
    relevant_texts = []
    for match in result['matches']:
        relevant_texts.append(match['metadata'].get('text', 'No content available'))
        source = match['metadata'].get('source', 'No source available')

        # relevant_texts.append(f"Source: {source}\n{relevant_texts}")
        # reference = "\n\n".join(relevant_texts)
        file = f"Source: {source}\n{relevant_texts}"
        print(f"{file}\n")
        template = '''
            롯데카드 is a South Korean credit card company, 로카 is short for 롯데카드, and LOCA is an acronym for Lotte Card. You are an online news management employee at 롯데카드. Your task is to find, summarize, and analyze news articles on the internet. The txt files uploaded contains the link, title, source, date, keyword and content of a news article.
            Please extract the following from files in order. Change the line for each category:
            - 제목: Title is the second line of the txt file. Include only the title.
            - 날짜: Date if the fourth line of the txt file. Write it in a format of yyyy-mm-dd.
            - 출처: Source is the third line of the txt file.
            - 키워드: Keyword is the fifth line of the txt file. If the keyword is between double quotes, extract without the double quotes.
            - 요약: Summarize the content of the article in 3 bullets. Specify in the summary if a specific company name is mentioned. Include in the summary if numbers are mentioned. Do not number the bullets. Do not use code blocks.
            - 링크: Link is the first line of the txt file. Please use a clear tone of voice.
            Please answer in Korean. 요약은 문장의 끝이 ~음., ~함. 과 같이 끝나는 음슴체로 작성해주세요.

            file: {file}
            answer: '''
        prompt = PromptTemplate.from_template(template)
        formatted_prompt = prompt.format(file=file)

        llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=openai_api_key)
        response = llm.invoke(formatted_prompt)
        print("LLM 답변: \n", response)

        # response = ollama.generate(model=ollama_model, prompt=formatted_prompt, options={'temperature': 0})
        # print("LLM 답변: \n", response['response'].strip())

if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd[0] == 'u':
        folder = '/svc/project/genaipilot/rag/data/news/'
        for file in os.listdir(folder):
            path = folder + file
            if os.path.isdir(path):
                continue
            else:
                print(path)
                embedding_text(path, 'news')
    elif cmd[0] == 'r':
        question = "뉴스 기사를 찾아줘"
        retrieve(question, 'news')