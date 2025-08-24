from simple_rag.index import IndexBuilder
from simple_rag.llm import BaseLLMProvider


class Pipeline:
    def __init__(self, index: IndexBuilder, llm: BaseLLMProvider, reranker=None):
        self.index = index
        self.llm = llm
        self.reranker = reranker

    def run(self, query, return_docs=False, max_tokens=150, temperature=0.7, top_k=5, stream=False):


        docs, distances = self.index(query, top_k=top_k)
        
        if self.reranker:
            docs = self.reranker(docs)

        context = "\n".join(docs)
        response = self.llm(query, context, max_tokens, temperature, stream)
        
        if return_docs:
            return response, (docs, distances)

        return response