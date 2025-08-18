from simple_rag.index import IndexBuilder
from simple_rag.llm import BaseLLMProvider


class Pipeline:
    def __init__(self, index: IndexBuilder, llm: BaseLLMProvider, reranker=None):
        self.index = index
        self.llm = llm
        self.reranker = reranker

    def run(self, query):
        docs, distances = self.index(query)
        
        if self.reranker:
            docs = self.reranker(docs)

        context = "\n".join(docs)
        response = self.llm(query, context)

        return response