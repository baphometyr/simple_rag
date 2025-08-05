import os
import faiss
import warnings

from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self, model_name:str='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).astype('float32')

    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()
           
class IndexBuilder:
    def __init__(self, embedding: Embedding, base_dir="indices"):
        self.embedding = embedding
        self.base_dir = base_dir
        self.texts = []
        self.index = None
        self.collection_name = None
        self.collection_path = None
        self.index_path = None
        self.texts_path = None

    def _set_paths(self, collection_name):
        self.collection_name = collection_name
        self.collection_path = os.path.join(self.base_dir, collection_name)
        self.index_path = os.path.join(self.collection_path, "index.faiss")
        self.texts_path = os.path.join(self.collection_path, "texts.txt")

    def load_index(self, path:str):
        self.index = faiss.read_index(path)

    def load_texts(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f.readlines()]

    def load_collection(self, collection_name):
        self._set_paths(collection_name)
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.texts_path):
            raise FileNotFoundError(f"Collection '{collection_name}' not found in '{self.collection_path}'")
        
        self.load_index(self.index_path)
        self.load_texts(self.texts_path)

    def save_index(self, path:str):
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        faiss.write_index(self.index, path)

    def save_texts(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for line in self.texts:
                f.write(f"{line}\n")
        
    def build_collection(self, collection_name, texts: list[str], use_advanced_index=False, nlist=100):
        self._set_paths(collection_name)
        os.makedirs(self.collection_path, exist_ok=True)

        self.texts = texts
        embeddings = self.embedding.encode(texts)
        dim = self.embedding.get_dimension()

        if use_advanced_index:
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(embeddings)
        else:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.save_index(self.index_path)
        self.save_texts(self.texts_path)

    def query(self, query: str, top_k: int = 5):
        if self.index is None:
            raise ValueError("Index not loaded or built.")

        if top_k > len(self.texts):
            warnings.warn(f"top_k ({top_k}) is greater than the number of texts ({len(self.texts)}). Setting top_k to {len(self.texts)}.")
            top_k = len(self.texts)
        
        query_embedding = self.embedding.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = []

        for i in range(top_k):
            idx = indices[0][i]
            dist = distances[0][i]
            text = self.texts[idx] if idx < len(self.texts) else None
            results.append((text, dist))
        return results
