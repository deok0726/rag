import os, sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

dotenv_path = os.path.join(os.path.dirname(__file__), 'config/.env')
load_dotenv(dotenv_path)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def retrieve_ensemble(context, question, index_name, namespace):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    query_vector = model.encode(question).tolist()
    result = index.query(namespace=namespace, vector=[query_vector], top_k=3, include_metadata=True)
    
    relevant_texts = []
    for match in result['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}")
        source = match['metadata'].get('source', 'No source available')
        texts = match['metadata'].get('text', 'No text available')
        relevant_texts.append(f"Source: {source}\n{texts}")

    reference = "\n\n".join(relevant_texts)
    print(reference)
    
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

    index_name = cmd[0]
    namespace = cmd[1]
    retrieve_ensemble(context[0], question[0], index_name, namespace)