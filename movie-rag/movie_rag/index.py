import os
import faiss
import warnings

from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer


class Embedding:
    """
    A wrapper for sentence-transformers to generate embeddings from text.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        """
        Initializes the embedding model.

        Args:
            model_name (str): The name of the sentence-transformers model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> 'np.ndarray':
        """
        Encodes a list of texts into vector embeddings.

        Args:
            texts (List[str]): A list of input strings.

        Returns:
            np.ndarray: A NumPy array of shape (len(texts), embedding_dim) with float32 vectors.
        """
        return self.model.encode(texts, convert_to_numpy=True).astype('float32')

    def get_dimension(self) -> int:
        """
        Returns the dimensionality of the embeddings produced by the model.

        Returns:
            int: The dimension of the sentence embeddings.
        """
        return self.model.get_sentence_embedding_dimension()


class IndexBuilder:
    """
    Builds, saves, loads, and queries FAISS indices using sentence embeddings.
    """

    def __init__(self, embedding: Embedding, base_dir: str = "indices") -> None:
        """
        Initializes the index builder.

        Args:
            embedding (Embedding): An instance of the Embedding class.
            base_dir (str): Directory where index collections are stored.
        """
        self.embedding = embedding
        self.base_dir = base_dir
        self.texts: List[str] = []
        self.index: Optional[faiss.Index] = None
        self.collection_name: Optional[str] = None
        self.collection_path: Optional[str] = None
        self.index_path: Optional[str] = None
        self.texts_path: Optional[str] = None

    def _set_paths(self, collection_name: str) -> None:
        """
        Sets the file paths for storing the index and text collection.

        Args:
            collection_name (str): Name of the collection.
        """
        self.collection_name = collection_name
        self.collection_path = os.path.join(self.base_dir, collection_name)
        self.index_path = os.path.join(self.collection_path, "index.faiss")
        self.texts_path = os.path.join(self.collection_path, "texts.txt")

    def load_index(self, path: str) -> None:
        """
        Loads a FAISS index from a given path.

        Args:
            path (str): File path to the FAISS index.
        """
        self.index = faiss.read_index(path)

    def load_texts(self, path: str) -> None:
        """
        Loads a list of texts from a file.

        Args:
            path (str): Path to the text file containing one text per line.
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f.readlines()]

    def load_collection(self, collection_name: str) -> None:
        """
        Loads an existing collection (index + texts) by name.

        Args:
            collection_name (str): Name of the collection to load.

        Raises:
            FileNotFoundError: If either the index or text file is missing.
        """
        self._set_paths(collection_name)

        if not os.path.exists(self.index_path) or not os.path.exists(self.texts_path):
            raise FileNotFoundError(f"Collection '{collection_name}' not found in '{self.collection_path}'")

        self.load_index(self.index_path)
        self.load_texts(self.texts_path)

    def save_index(self, path: str) -> None:
        """
        Saves the current FAISS index to a file.

        Args:
            path (str): Destination file path.
        """
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        faiss.write_index(self.index, path)

    def save_texts(self, path: str) -> None:
        """
        Saves the current list of texts to a file.

        Args:
            path (str): Destination text file path.
        """
        with open(path, 'w', encoding='utf-8') as f:
            for line in self.texts:
                f.write(f"{line}\n")

    def build_collection(
        self,
        collection_name: str,
        texts: List[str],
        use_advanced_index: bool = False,
        nlist: int = 100
    ) -> None:
        """
        Builds a new FAISS index from a list of texts.

        Args:
            collection_name (str): Name of the collection to build.
            texts (List[str]): List of texts to index.
            use_advanced_index (bool): Whether to use an IVF index (trained index) or not.
            nlist (int): Number of clusters for IVF index if used.
        """
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

    def query(self, query: str, top_k: int = 5) -> List[Tuple[Optional[str], float]]:
        """
        Searches the FAISS index for the top-k most similar texts to the query.

        Args:
            query (str): The input query string.
            top_k (int): The number of results to return.

        Returns:
            List[Tuple[Optional[str], float]]: A list of tuples with (matched_text, distance).
        """
        if self.index is None:
            raise ValueError("Index not loaded or built.")

        if top_k > len(self.texts):
            warnings.warn(
                f"top_k ({top_k}) is greater than the number of texts ({len(self.texts)}). Setting top_k to {len(self.texts)}."
            )
            top_k = len(self.texts)

        query_embedding = self.embedding.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results: List[Tuple[Optional[str], float]] = []

        for i in range(top_k):
            idx = indices[0][i]
            dist = distances[0][i]
            text = self.texts[idx] if idx < len(self.texts) else None
            results.append((text, dist))
        return results
