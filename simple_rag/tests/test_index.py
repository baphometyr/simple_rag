import os
import warnings
import pytest
from simple_rag.index import Embedding, IndexBuilder

EXAMPLE_TEXTS = [
    "The sun shines in the sky",
    "The moon is beautiful at night",
    "Cats are domestic animals",
    "Python is a programming language",
    "FAISS is used for vector searches"
]

# --- Fixtures ---

@pytest.fixture
def temp_dir(tmp_path):
    """Directorio temporal para guardar índices"""
    return tmp_path / "test_indices"

@pytest.fixture
def embedding():
    """Embedding para los tests"""
    return Embedding(model_name="all-MiniLM-L6-v2")

@pytest.fixture
def index_builder(embedding, temp_dir):
    """IndexBuilder base"""
    return IndexBuilder(embedding=embedding, base_dir=str(temp_dir))

@pytest.fixture
def built_index(index_builder):
    """Crea una colección lista para tests"""
    collection_name = "test_collection"
    index_builder.build_collection(collection_name, EXAMPLE_TEXTS)
    return index_builder

# --- Tests ---

def test_build_collection(built_index):
    # Verificar que los archivos fueron creados
    assert os.path.exists(built_index.index_path)
    assert os.path.exists(built_index.texts_path)
    # Verificar número de elementos
    assert built_index.index.ntotal == len(EXAMPLE_TEXTS)

def test_load_collection(embedding, temp_dir, built_index):
    # Cargar la colección desde disco
    new_builder = IndexBuilder.load_collection("test_collection", embedding, temp_dir)

    assert new_builder.texts == EXAMPLE_TEXTS
    assert new_builder.index.ntotal == len(EXAMPLE_TEXTS)

def test_query(built_index):
    query = "programming in Python"
    results = built_index.query(query, top_k=3)

    assert len(results) == 3
    # La respuesta más cercana debería ser "Python is a programming language"
    assert results[0][0] == "Python is a programming language"

def test_top_k_warning(built_index):
    # Solicitar más resultados de los que hay
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = built_index.query("cielo", top_k=10)

        # Se debe emitir advertencia
        assert any("top_k" in str(warning.message) for warning in w)
        # El número de resultados se ajusta al tamaño del texto
        assert len(results) == len(EXAMPLE_TEXTS)
