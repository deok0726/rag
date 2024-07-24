from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
elastic_vector_search = ElasticsearchStore(
    es_cloud_id="e30170eae6404b7092b537ad516a1d6e:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRkY2M5MjM3ZjY2OWU0NzRhYTk0OGQ5ZWVlMjU5NmRkMiQzYWUyZDE4MGMxMzA0NTQwYWY1MGFmYTVjZTIwODNkOA==",
    index_name="langchain_cloud",
    embedding=embedding,
    es_user="Austin Lee",
    es_password="elasticsearch0726"
)

