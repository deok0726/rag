from langchain_community.llms import Ollama

llm = Ollama(model="llama3:70b")
response = llm("What did the president say about Ketanji Brown Jackson")

print(response)