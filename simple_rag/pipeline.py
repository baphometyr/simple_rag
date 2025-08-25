from simple_rag.index import IndexBuilder
from simple_rag.llm import BaseLLMProvider


class Pipeline:
    def __init__(self, index: IndexBuilder, llm: BaseLLMProvider, reranker=None):
        if not index or not llm:
            raise ValueError("Index and LLM are required")

        self.index = index
        self.llm = llm
        self.reranker = reranker

    def run(
        self, 
        query:str,
        return_docs:bool=False, 
        max_tokens:int=150, 
        temperature:float=0.7, 
        top_k:int=5, 
        stream:bool=False
        ):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            docs, distances = self.index(query, top_k=top_k)
            
            # if self.reranker:
            #     docs = self.reranker(docs)

            context = "\n".join(docs)
            response = self.llm(query, context, max_tokens, temperature, stream)
        except Exception as e:
            raise RuntimeError(f"Pipeline execution failed: {str(e)}")

        
        if return_docs:
            return response, (docs, distances)

        return response