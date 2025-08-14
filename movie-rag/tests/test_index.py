import os
import shutil
import pytest
from movie_rag.index import Embedding, IndexBuilder

EXAMPLE_TEXTS = [
    "The sun shines in the sky",
    "The moon is beautiful at night",
    "Cats are domestic animals",
    "Python is a programming language",
    "FAISS is used for vector searches"
]

@pytest.fixture
def temp_dir(tmp_path):
    # temp directory for index testing
    return tmp_path / "test_indices"

@pytest.fixture
def embedding():
    return Embedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

@pytest.fixture
def index_builder(embedding, temp_dir):
    return IndexBuilder(embedding=embedding, base_dir=str(temp_dir))

def test_build_collection(index_builder):
    collection_name = "test_collection"
    index_builder.build_collection(collection_name, EXAMPLE_TEXTS)
    
    # Verificar que los archivos fueron creados
    assert os.path.exists(index_builder.index_path)
    assert os.path.exists(index_builder.texts_path)
    # Verificar que el índice tiene el número correcto de elementos
    assert index_builder.index.ntotal == len(EXAMPLE_TEXTS)

def test_load_collection(index_builder):
    collection_name = "test_collection"
    index_builder.build_collection(collection_name, EXAMPLE_TEXTS)
    
    # Crear un nuevo IndexBuilder para cargar
    new_builder = IndexBuilder(embedding=index_builder.embedding, base_dir=index_builder.base_dir)
    new_builder.load_collection(collection_name)
    
    assert new_builder.texts == EXAMPLE_TEXTS
    assert new_builder.index.ntotal == len(EXAMPLE_TEXTS)

def test_query(index_builder):
    collection_name = "test_collection"
    index_builder.build_collection(collection_name, EXAMPLE_TEXTS)
    
    query = "programación en Python"
    results = index_builder.query(query, top_k=3)
    
    # Verificar que se devuelven los resultados correctos
    assert len(results) == 3
    # La respuesta más cercana debería ser "Python es un lenguaje de programación"
    assert results[0][0] == "Python es un lenguaje de programación"
    # La distancia debe ser un float
    assert isinstance(results[0][1], float)

def test_top_k_warning(index_builder):
    collection_name = "test_collection"
    index_builder.build_collection(collection_name, EXAMPLE_TEXTS)
    
    # Solicitar más resultados que textos disponibles
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = index_builder.query("cielo", top_k=10)
        # Se debe emitir advertencia
        assert any("top_k" in str(warning.message) for warning in w)
        # El número de resultados se ajusta al tamaño del texto
        assert len(results) == len(EXAMPLE_TEXTS)
