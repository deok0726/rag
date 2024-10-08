import time
import os, sys
import ollama
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone, ServerlessSpec

dotenv_path = os.path.join(os.path.dirname(__file__), 'config/.env')
load_dotenv(dotenv_path)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def embedding_text(path, index_name, namespace):    
    loader = PDFPlumberLoader(path)
    print(f"Loaded PDF: {path}")
    data = loader.load()
    # print (f'You have {len(data)} document(s) in your data')

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # print (f'Now you have {len(docs)} documents')

    pc = Pinecone(api_key=pinecone_api_key)
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

    index.upsert(vectors=vectors_to_upsert, namespace=namespace)

    print(f'{index_name} upserted with documents: {file_name}')
    print(index.describe_index_stats())

def retrieve(context, question, index_name, namespace):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    # query = question.split(':')[-1]

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    # query_vector = model.encode(query).tolist()
    query_vector = model.encode(question).tolist()
    result = index.query(namespace=namespace, vector=[query_vector], top_k=3, include_metadata=True)
    
    relevant_texts = []
    for match in result['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}")
        # print(f"ID: {match['id']}, Score: {match['score']}, Text: {match['metadata']['text']}")

        # relevant_texts.append(match['metadata'].get('text', 'No content available'))
        source = match['metadata'].get('source', 'No source available')
        texts = match['metadata'].get('text', 'No text available')
        relevant_texts.append(f"Source: {source}\n{texts}")

    reference = "\n\n".join(relevant_texts)
    print(reference)
    few_shot_examples = '''
    ### context: 
    # 사실관계:
    - 롯데카드 Datus분석팀이 광고제휴사(롯데캐피탈, 여신전문금융업자)의 대출 상품(롯데카드 고객 대상 롯데캐피탈 대출상품)에 대하여 롯데카드의 LMS(Long Message Service) 광고 채널을 통해 광고제휴사로부터 대가를 받고 중개 및 주선하고자 함.
    - 사내 준법경영팀과 해당 유료광고 기획 건들에 대한 진행 가능 여부 및 고려사항에 대해 논의하고, 롯데카드의 사업자 신고 업종 中 '대출의 중개 및 주선업무' 항목을 근거로, 해당 유료 광고 기획 건들을 추진할 수 있는지 법무팀에게 법무검토를 받고자 함.
    - 현재까지 타사의 대출상품에 대해 롯데카드의 LMS 광고 채널을 통해서 유료 광고를 진행한 이력은 없음.
    - 현재까지 타사(핀다, 전자금융업종)의 대출대환서비스를 여신전문금융업종에 속한 타사(신한카드)가 LMS 광고 채널을 통해 광고를 진행한 이력은 있음.

    ### reference: 
    # 관련 법령의 전문
    여신전문금융업법 제14조(신용카드·직불카드의 발급)
    ④ 신용카드업자는 다음 각 호의 방법으로 신용카드회원을 모집하여서는 아니 된다.
    「방문판매 등에 관한 법률」 제2조 제5호에 따른 다단계판매를 통한 모집
    인터넷을 통한 모집방법으로서 대통령령으로 정하는 모집
    그 밖에 대통령령으로 정하는 모집
    여신전문금융업법 제14조의5(모집질서의 유지)
    ③ 신용카드회원을 모집하는 자는 제14조 제4항 각 호의 행위 및 제24조의2(신용카드회원 모집행위와 관련된 행위에 한한다)에 따른 금지행위를 하여서는 아니 된다.
    여신전문금융업법 시행령 제6조의7(신용카드의 발급 및 회원 모집방법 등)
    ⑤ 법 제14조 제4항 제3호에 따라 신용카드업자는 다음 각 호의 방법으로 신용카드회원을 모집해서는 아니 된다.
    신용카드 발급과 관련하여 그 신용카드 연회비(연회비가 주요 신용카드의 평균연회비 미만인 경우에는 해당 평균연회비를 말한다)의 100분의 10을 초과하는 경제적 이익을 제공하거나 제공할 것을 조건으로 하는 모집
    「도로법」 제2조 및 「사도법」 제2조에 따른 도로 및 사도 등 길거리에서 하는 모집
    
    # 관련 판례 요약
    헌법재판소 2013. 3. 26. 선고 2013헌마110 결정
    사건: 2013헌마110 여신전문금융업법 제14조 제4항 제3호 등 위헌확인 청구인들은 신용카드 모집인이었으며, 신용카드 모집 과정에서 연회비의 10%를 초과하는 혜택을 제공하거나 공공장소에서 모집을 진행한 행위로 과태료 처분을 받은 후 해당 조항들이 기본권을 침해한다고 주장하면서 위헌 확인을 요청하였으나, 헌법재판소는 청구인들의 헌법소원 심판 청구가 부적법하다며 각하하였다.

    ### question:
    # 질의의 요지:
    - 롯데카드 광고채널(LMS)을 통한 '타사(롯데캐피탈)의 대출상품 중개 및 주선업무' 가능 여부
    - 준법경영팀과 논의한 결과, 특정 고객군을 타겟팅하여 대출광고를 진행하는 경우, 단순 광고가 아닌 중개업무로 해석될 여지가 있으므로 당사의 사업자 신고업종 中 '대출상품 중개 및 주선업무' 항목을 근거로 당사에서 타사(롯데캐피탈)의 대출상품 중개 및 주선업무가 가능한지에 대해 법적으로 확인할 것을 권고함. 
    
    ### answer: 
    # 질의에 대한 답변
    1. 타사(롯데캐피탈)의 대출상품 중개 및 주선업무 가능 여부
    해당 사례에서 롯데카드가 LMS 광고 채널을 통해 타사의 대출상품(롯데캐피탈)을 중개 및 주선하는 것이 가능한지에 대한 질의는 여신전문금융업법과 관련된 규정과 그 적용 범위에 근거하여 검토할 필요가 있습니다.
    롯데카드는 "여신전문금융업법"상 신용카드업자로서, 신용카드 모집 및 광고 활동과 관련하여 엄격한 규제를 받고 있습니다. 특히 여신전문금융업법 제14조는 신용카드 모집 방법에 대해 제한을 두고 있으며, 신용카드 모집인 또는 신용카드업자가 대출상품을 광고하는 과정에서 법적 한계가 있을 수 있습니다. 여신전문금융업법 시행령 제6조의7에서는 신용카드 모집 시 제공할 수 있는 혜택과 모집 방법을 명시적으로 규정하고 있으며, 이러한 규정을 위반할 경우 과태료 또는 처벌을 받을 수 있습니다.
    이와 관련된 헌법재판소 2013헌마110 결정에서도, 모집인에 대한 규제 사항들이 명확히 적용된 사례를 확인할 수 있습니다. 이 결정에서는 모집 과정에서 허용된 범위를 넘어설 경우 과태료 처분이 내려질 수 있으며, 이러한 규제가 정당하다는 결론을 내리고 있습니다.
    따라서, 롯데카드가 타사의 대출상품을 광고하는 과정에서 LMS 광고 채널을 통해 광고를 진행하는 것은, 그 범위 내에서 단순한 광고를 넘어서는 중개 및 주선 업무로 해석될 가능성이 있습니다. 롯데카드의 사업자 신고 업종에 '대출상품 중개 및 주선업무'가 포함되어 있다 하더라도, 여신전문금융업법 및 관련 시행령의 적용을 받을 수 있으며, 법적으로 허용된 광고 방법 이외의 방식으로 대출 상품을 중개하거나 주선하는 행위는 법적 리스크가 있을 수 있습니다.
    
    2. 결론 및 법적 고려사항
    결론적으로, 롯데카드가 LMS 광고 채널을 통해 타사(롯데캐피탈)의 대출상품을 중개 및 주선하는 행위는 여신전문금융업법의 규제 범위 내에서 신중하게 검토되어야 합니다. 준법경영팀의 권고대로 해당 광고가 단순 광고로 해석되지 않고 중개 업무로 해석될 여지가 있으므로, 광고 방식, 대가 수취 여부, 구체적인 중개 과정 등을 고려하여 법적 리스크가 발생할 가능성이 있습니다.
    이에 따라, 관련 법령 및 판례를 바탕으로 중개 및 주선 행위와 단순 광고 행위의 구분이 명확하게 이뤄져야 하며, 중개로 판단될 경우 추가적인 법적 검토와 절차가 필요할 것입니다.
    '''
    template = '''
        You're a financial legal expert. Please answer the questions based on the laws, terms, articles, government policies, cases and contexts below.
        
        The format for your response is as follows
        - Full text of the reference: Specify the name and the sections of the reference you need to refer to for your answer, and print out the full text of the reference exactly as it is. Prioritize finding laws and case law for reference. Do not summarize anything. Include everything in parentheses. Accuracy is very important. If there are multiple references for your answer, please find and print them all. Find as many references as possible.
        - Answer to the question: Write a response to the '질의의 요지' based on the sections you referenced. Please explain in detail the reasoning behind your answer.
        Use a clear tone of voice. Please answer in Korean.
        Think step by step.

        context: 
            {context}
        question:
            {question}
        reference: 
            {reference}

        answer: '''
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(reference=reference, context=context, question=question)
    
    # ollama_model = 'llama3.1:70b'
    # response = ollama.generate(model=ollama_model, prompt=formatted_prompt, options={'temperature': 0})
    # print("LLM 답변: \n", response['response'].strip())

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=openai_api_key)
    response = llm.invoke(formatted_prompt)
    print("LLM 답변: \n", response.content)

def retrieve_all(context, question, index_name):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    query_vector = model.encode(question).tolist()
        
    index_stats = index.describe_index_stats()
    all_namespaces = index_stats['namespaces'].keys()
    
    top_results = []
    relevant_texts = []
    for namespace in all_namespaces:
        print(f"Searching in namespace: {namespace}")
        
        result = index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )

        if result['matches']:
            for match in result['matches']:
                if match['score'] >= 0.6:
                    match['namespace'] = namespace
                    top_results.append(match)

        # if result['matches']:
        #     best_match = result['matches'][0]
        #     if best_match['score'] >= 0.6:
        #         best_match['namespace'] = namespace
        #         top_results.append(best_match)

    if top_results:
        for result in top_results:
            print(f"Namespace: {result['namespace']}")
            print(f"ID: {result['id']}, Score: {result['score']}")
            # print(f"ID: {result['id']}, Score: {result['score']}, Metadata: {result.get('metadata', {})}")
            
            namespace = result['namespace']
            source = result['metadata'].get('source', 'No source available')
            texts = result['metadata'].get('text', 'No text available')
            relevant_texts.append(f"Namespace: {namespace}\nSource: {source}\n{texts}")
    else:
        print("No matches with score >= 0.6 found in any namespace.")

    reference = "\n\n".join(relevant_texts)
    # print(reference)
    few_shot_examples = '''
    ### context: 
    # 사실관계:
    - 롯데카드 Datus분석팀이 광고제휴사(롯데캐피탈, 여신전문금융업자)의 대출 상품(롯데카드 고객 대상 롯데캐피탈 대출상품)에 대하여 롯데카드의 LMS(Long Message Service) 광고 채널을 통해 광고제휴사로부터 대가를 받고 중개 및 주선하고자 함.
    - 사내 준법경영팀과 해당 유료광고 기획 건들에 대한 진행 가능 여부 및 고려사항에 대해 논의하고, 롯데카드의 사업자 신고 업종 中 '대출의 중개 및 주선업무' 항목을 근거로, 해당 유료 광고 기획 건들을 추진할 수 있는지 법무팀에게 법무검토를 받고자 함.
    - 현재까지 타사의 대출상품에 대해 롯데카드의 LMS 광고 채널을 통해서 유료 광고를 진행한 이력은 없음.
    - 현재까지 타사(핀다, 전자금융업종)의 대출대환서비스를 여신전문금융업종에 속한 타사(신한카드)가 LMS 광고 채널을 통해 광고를 진행한 이력은 있음.

    ### reference: 
    # 관련 법령의 전문
    여신전문금융업법 제14조(신용카드·직불카드의 발급)
    ④ 신용카드업자는 다음 각 호의 방법으로 신용카드회원을 모집하여서는 아니 된다.
    「방문판매 등에 관한 법률」 제2조 제5호에 따른 다단계판매를 통한 모집
    인터넷을 통한 모집방법으로서 대통령령으로 정하는 모집
    그 밖에 대통령령으로 정하는 모집
    여신전문금융업법 제14조의5(모집질서의 유지)
    ③ 신용카드회원을 모집하는 자는 제14조 제4항 각 호의 행위 및 제24조의2(신용카드회원 모집행위와 관련된 행위에 한한다)에 따른 금지행위를 하여서는 아니 된다.
    여신전문금융업법 시행령 제6조의7(신용카드의 발급 및 회원 모집방법 등)
    ⑤ 법 제14조 제4항 제3호에 따라 신용카드업자는 다음 각 호의 방법으로 신용카드회원을 모집해서는 아니 된다.
    신용카드 발급과 관련하여 그 신용카드 연회비(연회비가 주요 신용카드의 평균연회비 미만인 경우에는 해당 평균연회비를 말한다)의 100분의 10을 초과하는 경제적 이익을 제공하거나 제공할 것을 조건으로 하는 모집
    「도로법」 제2조 및 「사도법」 제2조에 따른 도로 및 사도 등 길거리에서 하는 모집
    
    # 관련 판례 요약
    헌법재판소 2013. 3. 26. 선고 2013헌마110 결정
    사건: 2013헌마110 여신전문금융업법 제14조 제4항 제3호 등 위헌확인 청구인들은 신용카드 모집인이었으며, 신용카드 모집 과정에서 연회비의 10%를 초과하는 혜택을 제공하거나 공공장소에서 모집을 진행한 행위로 과태료 처분을 받은 후 해당 조항들이 기본권을 침해한다고 주장하면서 위헌 확인을 요청하였으나, 헌법재판소는 청구인들의 헌법소원 심판 청구가 부적법하다며 각하하였다.

    ### question:
    # 질의의 요지:
    - 롯데카드 광고채널(LMS)을 통한 '타사(롯데캐피탈)의 대출상품 중개 및 주선업무' 가능 여부
    - 준법경영팀과 논의한 결과, 특정 고객군을 타겟팅하여 대출광고를 진행하는 경우, 단순 광고가 아닌 중개업무로 해석될 여지가 있으므로 당사의 사업자 신고업종 中 '대출상품 중개 및 주선업무' 항목을 근거로 당사에서 타사(롯데캐피탈)의 대출상품 중개 및 주선업무가 가능한지에 대해 법적으로 확인할 것을 권고함. 
    
    ### answer: 
    # 질의에 대한 답변
    1. 타사(롯데캐피탈)의 대출상품 중개 및 주선업무 가능 여부
    해당 사례에서 롯데카드가 LMS 광고 채널을 통해 타사의 대출상품(롯데캐피탈)을 중개 및 주선하는 것이 가능한지에 대한 질의는 여신전문금융업법과 관련된 규정과 그 적용 범위에 근거하여 검토할 필요가 있습니다.
    롯데카드는 "여신전문금융업법"상 신용카드업자로서, 신용카드 모집 및 광고 활동과 관련하여 엄격한 규제를 받고 있습니다. 특히 여신전문금융업법 제14조는 신용카드 모집 방법에 대해 제한을 두고 있으며, 신용카드 모집인 또는 신용카드업자가 대출상품을 광고하는 과정에서 법적 한계가 있을 수 있습니다. 여신전문금융업법 시행령 제6조의7에서는 신용카드 모집 시 제공할 수 있는 혜택과 모집 방법을 명시적으로 규정하고 있으며, 이러한 규정을 위반할 경우 과태료 또는 처벌을 받을 수 있습니다.
    이와 관련된 헌법재판소 2013헌마110 결정에서도, 모집인에 대한 규제 사항들이 명확히 적용된 사례를 확인할 수 있습니다. 이 결정에서는 모집 과정에서 허용된 범위를 넘어설 경우 과태료 처분이 내려질 수 있으며, 이러한 규제가 정당하다는 결론을 내리고 있습니다.
    따라서, 롯데카드가 타사의 대출상품을 광고하는 과정에서 LMS 광고 채널을 통해 광고를 진행하는 것은, 그 범위 내에서 단순한 광고를 넘어서는 중개 및 주선 업무로 해석될 가능성이 있습니다. 롯데카드의 사업자 신고 업종에 '대출상품 중개 및 주선업무'가 포함되어 있다 하더라도, 여신전문금융업법 및 관련 시행령의 적용을 받을 수 있으며, 법적으로 허용된 광고 방법 이외의 방식으로 대출 상품을 중개하거나 주선하는 행위는 법적 리스크가 있을 수 있습니다.
    
    2. 결론 및 법적 고려사항
    결론적으로, 롯데카드가 LMS 광고 채널을 통해 타사(롯데캐피탈)의 대출상품을 중개 및 주선하는 행위는 여신전문금융업법의 규제 범위 내에서 신중하게 검토되어야 합니다. 준법경영팀의 권고대로 해당 광고가 단순 광고로 해석되지 않고 중개 업무로 해석될 여지가 있으므로, 광고 방식, 대가 수취 여부, 구체적인 중개 과정 등을 고려하여 법적 리스크가 발생할 가능성이 있습니다.
    이에 따라, 관련 법령 및 판례를 바탕으로 중개 및 주선 행위와 단순 광고 행위의 구분이 명확하게 이뤄져야 하며, 중개로 판단될 경우 추가적인 법적 검토와 절차가 필요할 것입니다.
    '''
    template = '''
        You're a financial legal expert. Please answer the questions based on the laws, terms, precedents and context of question below.
        Use a clear tone of voice. Please answer in Korean. Think step by step.

        The format for your response is as follows
        - Source and full text of the law referenced: Specify the sections of the reference you need to refer to for your answer, and print out the full text of the reference exactly as it is. Do not summarize anything. Include everything in parentheses. Accuracy is very important. If there are multiple cases that you need to reference for your answer, please find and print them all. Find as many relevant cases as possible.
        - Source and full text of the terms referenced: Specify the sections of the reference you need to refer to for your answer, and print out the full text of the reference exactly as it is. Do not summarize anything. Include everything in parentheses. Accuracy is very important. If there are multiple cases that you need to reference for your answer, please find and print them all. Find as many relevant cases as possible. If there is no reference to refer to, do not print anything.
        - Source and full text of the precedents referenced: Specify the sections of the case you need to refer to for your answer, and print out the full text of the reference exactly as it is. Do not summarize anything. Include everything in parentheses. Accuracy is very important. If there are multiple sections that you need to reference for your answer, please find and print them all. Find as many relevant sections as possible. If there is no reference to refer to, do not print anything.

        context: 
            {context}
        question:
            {question}
        reference: 
            {reference}

        answer: '''
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(reference=reference, context=context, question=question)
    print(formatted_prompt)
    
    # ollama_model = 'llama3.1:70b'
    # response = ollama.generate(model=ollama_model, prompt=formatted_prompt, options={'temperature': 0})
    # print("LLM 답변: \n", response['response'].strip())

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, openai_api_key=openai_api_key)
    response = llm.invoke(formatted_prompt)
    print("LLM 답변: \n", response.content)

def test(query, index_name, namespace):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    embedding = model.encode(query).tolist()  # query 벡터 생성
    results = index.query(
        namespace=namespace,
        vector=embedding,
        top_k=3,
        include_values=False,
        include_metadata=True
    )

    for match in results['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}, Text: {match['metadata']['text']}")

if __name__ == "__main__":
    context = [
    """사실관계:
    - CLP 신용카드 유치 시 불특정 다수에게 과다경품(현금) 제공 홍보하였으나, 실제 카드유치가 되지 않고, 행위에 대한 결과가 없을때 해당행위가 정도영업에 위반인지 여부 판단.
    - 회원모집 업무 위탁 계약서:
    - 준수사항 1. 회원유치 시 불법 판촉물 제공, 길거리영업, 이중영업, 연회비 대납 등 정도영업 위반행위를 절대 하지 아니한다.""",
    """사실관계:
    - 롯데카드 Datus분석팀이 광고제휴사(롯데캐피탈, 여신전문금융업자)의 대출 상품(롯데카드 고객 대상 롯데캐피탈 대출상품)에 대하여 롯데카드의 LMS(Long Message Service) 광고 채널을 통해 광고제휴사로부터 대가를 받고 중개 및 주선하고자 함.
    - 사내 준법경영팀과 해당 유료광고 기획 건들에 대한 진행 가능 여부 및 고려사항에 대해 논의하고, 롯데카드의 사업자 신고 업종 中 '대출의 중개 및 주선업무' 항목을 근거로, 해당 유료 광고 기획 건들을 추진할 수 있는지 법무팀에게 법무검토를 받고자 함.
    - 현재까지 타사의 대출상품에 대해 롯데카드의 LMS 광고 채널을 통해서 유료 광고를 진행한 이력은 없음.
    - 현재까지 타사(핀다, 전자금융업종)의 대출대환서비스를 여신전문금융업종에 속한 타사(신한카드)가 LMS 광고 채널을 통해 광고를 진행한 이력은 있음.""",
    """사실관계:
    - 2023.10.24 최초 연체됨
    - 회원이 운영하는 가맹점 3곳 확인됨(헬스트레이닝관련)-개인사업자    1번가맹점(2021.09.29계약/2022.10.25폐업), 2번가맹점(2021.05.13계약/2023.10.17폐업), 3번가맹점-공동대표(2020.05.07계약/운영중)
    - 회원 본인 명의의 신용카드로 3번가맹점에서 과거부터 매출발생됨""",
    """사실관계:
    - 당팀 제휴카드 LG전자 스페셜 롯데카드의 서비스인 LG렌탈료 결제일 할인 서비스를 카드유효기간 內 60개월로 한정 됨. LG전자 렌탈 제품 판매 시, 고객과 계약(의무 사용 기간 약정)을 72개월로 판매함에 따라 당사 제휴카드 할인 서비스를 72개월 간 적용하기 위해 유효기간을 5년 → 10년으로 연장 하고자 함""",
    """사실관계:
    - 금감원은 온라인 설명의무 가이드라인('22년 8월) 적용을 위한 시스템 구축 계획 파악 요청함
    - 주요내용:
    - 파트1~파트7까지 온라인 설명의무 가이드라인 구성됨
    - 금융소비자 대상 온라인 판매시 금융소비자가 금융상품에 대해 쉽고/명확하게 인식할 수 있도록
    - 금융사는 화면구성/정보탐색/상담채널구비/금융소비자 불이익사항 등을 명확하게 표시해야 함
    - 당사 대응
    - 온라인상 상품 판매중인 팀 대상으로 해당 가이드라인을 기초로 자체점검 실시 (CL사업팀, 종합금융지원팀, 디지털영업팀, 법인기획팀, 카드Biz팀)
    - 판매 상품
    - CL사업팀 : 장/단기카드대출, 사업자신용대출
    - 종합금융지원팀 : 내구재, 스탁론
    - 디지털영업팀 : 신용카드
    - 카드Biz팀 : 홈페이지, 모바일 웹 구축
    - 법인기획팀 : 법인카드
    ※ 법인카드의 경우 안내 대상이 되는자가 모호함. 발급/신청/한도부여/실 사용 등 업무 처리에 따라 대상이 다름"""
    ]
    question = [
        """질의의 요지:
        - 여전법과 회원모집 업무 위탁 계약서 내 과다경품(불법판촉물, 연회비 대납 등) 위반시 정도영업 위반이라는 항목이 있으나, 불특정 다수에게 해당 행위를 구두로 홍보하였으나, 실제 행위의 대상이 없고 카드유치건이 없어도 정도영업 위반에 해당 되는지, 실제 결과 값이 아닌 행위만으로 여전법(과다경품제공)에 저촉이 되는지 검토요청 - 내/외부 점검반 모집현장(유치부스) 점검시 일부 CLP(모집인) 부스에서 불특정 다수에게 과다경품을 드리오니 카드신청하세요라는 영상을 촬영하여 과다경품지급 위반으로 당사 내부 정도영업 심의 회부되는 사례가 발생. 과다경품 제공조건으로 카드를 만들라며 구두로 불특정 다수에게 안내하였음. 당일 해당 CLP(모집인)는 유치건이 없으며, 증빙 영상 내 실제 과다경품 행위를 하는 내용이나 대상자는 없음. 위 내용을 바탕으로 결과없이 행위만으로도 법률에 저촉이 되고, 처벌이 가능한지 검토 요청드립니다.""",
        """질의의 요지:
        - 롯데카드 광고채널(LMS)을 통한 '타사(롯데캐피탈)의 대출상품 중개 및 주선업무' 가능 여부
        - 준법경영팀과 논의한 결과, 특정 고객군을 타겟팅하여 대출광고를 진행하는 경우, 단순 광고가 아닌 중개업무로 해석될 여지가 있으므로 당사의 사업자 신고업종 中 '대출상품 중개 및 주선업무' 항목을 근거로 당사에서 타사(롯데캐피탈)의 대출상품 중개 및 주선업무가 가능한지에 대해 법적으로 확인할 것을 권고함.""",
        """질의의 요지:
        - 공동대표(개인사업자)로 되어있긴 하나 회원이 대표로 되어있는 사업장에서 회원 명의의 신용카드로 매출을 일으킨것에 대한 형법상 적용가능 죄목(일부상환/개인회생을 목적으로 현재 부채증명서발급)
        - 형법상 적용가능 죄목이 있다면, 회원외 주대표로 되어 있는 공동대표에게도 적용 가능한지 여부""",
        """질의의 요지:
        - LG전자 스페셜 롯데카드의 유효기간을 5년 → 10년으로 연장하여 운영할 경우 법적으로 문제가 되는 부분이 있는지 검토 요청 드립니다.""",
        """질의의 요지:
        - 본 온라인 설명의무 가이드라인에서 지칭하고 있는 금융소비자의 명확한 정의가 무엇인지? (법인/개인/사업자 등 금융상품을 이용하고 있는 모든 소비자를 뜻하는지?)"""
    ]
    cmd = sys.argv[1:]
    namespace = cmd[1]
    if cmd[0] == 'u':
        folder = f'/svc/project/genaipilot/rag/data/legal/{namespace}/'
        for file in os.listdir(folder):
            path = folder + file
            if os.path.isdir(path):
                continue
            else:
                print(path)
                embedding_text(path, 'legal', namespace)
    elif cmd[0] == 'r':
        # question = input("질문을 입력하세요: ")
        # context = '''
        # 사실관계:
        # - CLP 신용카드 유치 시 불특정 다수에게 과다경품(현금) 제공 홍보하였으나, 실제 카드유치가 되지 않고, 행위에 대한 결과가 없을때 해당행위가 정도영업에 위반인지 여부 판단.
        # - 회원모집 업무 위탁 계약서:
        # - 준수사항 1. 회원유치 시 불법 판촉물 제공, 길거리영업, 이중영업, 연회비 대납 등 정도영업 위반행위를 절대 하지 아니한다.
        # '''
        # question = '''
        # 질의의 요지:
        # - 여전법과 회원모집 업무 위탁 계약서 내 과다경품(불법판촉물, 연회비 대납 등) 위반시 정도영업 위반이라는 항목이 있으나, 불특정 다수에게 해당 행위를 구두로 홍보하였으나, 실제 행위의 대상이 없고 카드유치건이 없어도 정도영업 위반에 해당 되는지, 실제 결과 값이 아닌 행위만으로 여전법(과다경품제공)에 저촉이 되는지 검토요청 - 내/외부 점검반 모집현장(유치부스) 점검시 일부 CLP(모집인) 부스에서 불특정 다수에게 과다경품을 드리오니 카드신청하세요라는 영상을 촬영하여 과다경품지급 위반으로 당사 내부 정도영업 심의 회부되는 사례가 발생. 과다경품 제공조건으로 카드를 만들라며 구두로 불특정 다수에게 안내하였음. 당일 해당 CLP(모집인)는 유치건이 없으며, 증빙 영상 내 실제 과다경품 행위를 하는 내용이나 대상자는 없음. 위 내용을 바탕으로 결과없이 행위만으로도 법률에 저촉이 되고, 처벌이 가능한지 검토 요청드립니다.
        # '''
        if namespace == 'all':
            for i in range(5):
                retrieve_all(context[i], question[i], 'legal')
        else:
            for i in range(5):
                # print(context[i])
                # print(question[i])
                retrieve(context[i], question[i], 'legal', namespace)
    elif cmd[0] == 't':
        # query = "금융소비자에 대한 내용이 담긴 법률을 알려줘"
        query = "마약 관련 법안"
        test(query, 'legal', namespace)
    elif cmd[0] == 'p':
        answer = "여신전문금융업법 제15조 제4항: (업법 제15조 제4항이 신용카드가맹점의 준수사항의 하나로서 신용카드가맹점은 물품의 판매 또는 용역의 제공이 없이 신용카드에 의한 거래를 한 것으로 가장하여 매출전표를 작성하는 행위를 하여서는 아니된다고 규정하고 있는 점에 비추어 보면, 피고인이 이와 같은 허위의 매출전표에 의한 대금청구에 대하여는 신용카드회사가 그 대금지급을 거절할 수 있음이 추단된다고 할 것이다) 그 판시와 같은 이유만으로 이 사건 예비적 공소사실을 무죄로 판단하고 말았으니, 이러한 원심의 판단에는 상고이유에서 지적하는 바와 같이 사기죄의 편취범의에 관한 법리를 오해하여 심리를 다하지 아니한 잘못이 있다고 할 것이다. 상고이유 중 이 점을 지적하는 부분은 이유 있다.\n  - 신용정보의 이용 및 보호에 관한 법률 제18조제1항, 제20조의2제1항ㆍ제3항 또는 제4항, 제22조의2, 제22조의6제4항, 제27조제8항, 제31조, 제32조제3항ㆍ제7항 또는 제10항, 제35조, 제35조의2, 제40조의2제8항, 제41조의2제3항: (신용정보의 이용 및 보호에 관한 법률 5. 제18조제1항을 위반한 자 6. 제20조의2제1항ㆍ제3항 또는 제4항을 위반한 자 7. 제22조의2를 위반하여 금융위원회에 보고를 하지 아니한 자 7의2. 제22조의6제4항을 위반하여 이용자관리규정을 정하지 아니한 자 8. 제27조제8항을 위반하여 채권추심업무를 할 때 증표를 내보이지 아니한 자 9. 제31조를 위반한 자 10. 제32조제3항ㆍ제7항 또는 제10항(제34조에 따라 준용하는 경우를 포함한다)을 위반한 자 11. 제35조를 위반한 자 11의2. 제35조의2를 위반하여 해당 신용정보주체에게 설명하지 아니한 자 11의3. 제40조의2제8항을 위반하여 개인신용정보를 가명처리하거나 익명처리한 기록을 보존하지 아니한 자 12. 제41조의2제3항을 위반하여 위탁계약 해지에 관한 사항을 알리지 아니한 자)\n\n- Answer to the question:\n  질의의 요지에 대한 답변을 위해서는 여신전문금융업법과 신용정보의 이용 및 보호에 관한 법률의 관련 조항을 검토해야 합니다. \n\n  여신전문금융업법 제15조 제4항은 신용카드가맹점이 물품의 판매 또는 용역의 제공 없이 신용카드 거래를 가장하여 매출전표를 작성하는 행위를 금지하고 있습니다. 이는 실제 거래가 없는 상황에서 허위로 매출을 발생시키는 행위를 방지하기 위한 규정입니다. \n\n  질의의 경우, CLP(모집인)가 불특정 다수에게 과다경품을 제공하겠다는 구두 홍보를 하였으나, 실제로 카드 유치가 이루어지지 않았고, 과다경품이 제공되지 않았습니다. 따라서, 실제로 과다경품이 제공되지 않았고, 카드 유치가 이루어지지 않은 상황에서는 여신전문금융업법 제15조 제4항의 위반으로 보기 어렵습니다. \n\n  또한, 신용정보의 이용 및 보호에 관한 법률의 관련 조항을 검토한 결과, 과다경품 제공과 관련된 직접적인 규정은 발견되지 않았습니다. 따라서, 실제 행위가 이루어지지 않은 상황에서 법률 위반으로 처벌하기는 어려울 것으로 판단됩니다.\n\n  그러나, 회사 내부의 정도영업 규정에 따라 과다경품 제공을 홍보한 행위 자체가 내부 규정 위반으로 판단될 수 있으며, 이에 대한 내부 징계나 조치가 이루어질 수 있습니다. 법률적 처벌과는 별개로, 회사의 내부 규정에 따른 조치가 필요할 수 있습니다. \n\n  결론적으로, 실제 결과가 없는 행위만으로 여신전문금융업법이나 신용정보의 이용 및 보호에 관한 법률에 저촉되어 법률적 처벌을 받기는 어려우나, 내부 규정 위반으로 인한 내부 조치는 가능할 수 있습니다."
        print(answer)