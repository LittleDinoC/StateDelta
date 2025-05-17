from transformers import AutoTokenizer
from src.retriever.retriever import BM25

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
bm25_retriever = BM25(
    tokenizer = tokenizer, 
    index_name = "wiki", 
    engine = "elasticsearch",
)

def bm25_retrieve(question, topk):
    docs_ids, docs = bm25_retriever.retrieve(
        [question], 
        topk=topk, 
        max_query_length=256
    )
    return docs[0].tolist()