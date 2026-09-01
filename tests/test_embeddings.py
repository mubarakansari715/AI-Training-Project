from app.rag.embeddings import EmbeddingService


def test_embed_texts_returns_one_vector_per_text():
    service = EmbeddingService()
    vectors = service.embed_texts(["hello world", "annual leave policy"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0


def test_embed_texts_empty_list_returns_empty_list():
    service = EmbeddingService()
    assert service.embed_texts([]) == []


def test_embed_query_returns_a_single_vector():
    service = EmbeddingService()
    vector = service.embed_query("What is the leave policy?")

    assert isinstance(vector, list)
    assert len(vector) > 0


def test_model_is_cached_across_instances():
    service_a = EmbeddingService()
    service_b = EmbeddingService()

    # Both instances should share the same underlying cached model.
    assert service_a._model is service_b._model
