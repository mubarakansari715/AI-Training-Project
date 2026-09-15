import pytest

from app.rag.retriever import RetrievedChunk
from app.services.conversation_service import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(storage_dir=str(tmp_path / "conversations"))


def _messages(with_sources: bool = False):
    sources = (
        [RetrievedChunk(text="Some text.", source="handbook.pdf", page=1, score=0.5, chunk_id="c1")]
        if with_sources
        else []
    )
    return [
        {"role": "user", "content": "What is the leave policy?"},
        {
            "role": "assistant",
            "content": "You get 24 days.",
            "sources": sources,
            "source": "document",
            "show_debug": False,
        },
    ]


def test_save_and_load_round_trips_messages(store):
    conversation_id = store.new_id()
    store.save(conversation_id, _messages())

    loaded = store.load(conversation_id)

    assert loaded[0] == {
        "role": "user",
        "content": "What is the leave policy?",
        "sources": [],
        "source": None,
        "show_debug": False,
        "attached_files": [],
    }
    assert loaded[1]["content"] == "You get 24 days."
    assert loaded[1]["source"] == "document"


def test_save_and_load_round_trips_sources(store):
    conversation_id = store.new_id()
    store.save(conversation_id, _messages(with_sources=True))

    loaded = store.load(conversation_id)

    assert loaded[1]["sources"] == [
        RetrievedChunk(text="Some text.", source="handbook.pdf", page=1, score=0.5, chunk_id="c1")
    ]


def test_save_and_load_round_trips_attached_files(store):
    conversation_id = store.new_id()
    messages = [
        {
            "role": "user",
            "content": "",
            "attached_files": ["handbook.pdf", "policy.txt"],
        },
        {"role": "assistant", "content": "Indexed 2 files."},
    ]
    store.save(conversation_id, messages)

    loaded = store.load(conversation_id)

    assert loaded[0]["attached_files"] == ["handbook.pdf", "policy.txt"]


def test_save_with_no_messages_does_not_create_a_file(store):
    conversation_id = store.new_id()
    store.save(conversation_id, [])

    assert store.load(conversation_id) == []
    assert store.list_conversations() == []


def test_load_missing_conversation_returns_empty_list(store):
    assert store.load("does-not-exist") == []


def test_title_is_auto_generated_from_first_user_message(store):
    conversation_id = store.new_id()
    store.save(conversation_id, _messages())

    conversations = store.list_conversations()

    assert conversations[0]["title"] == "What is the leave policy?"


def test_long_title_is_truncated(store):
    conversation_id = store.new_id()
    long_question = "A" * 100
    store.save(conversation_id, [{"role": "user", "content": long_question}])

    conversations = store.list_conversations()

    assert len(conversations[0]["title"]) <= 51  # 50 chars + ellipsis
    assert conversations[0]["title"].endswith("…")


def test_list_conversations_sorted_newest_first(store):
    first_id = store.new_id()
    store.save(first_id, [{"role": "user", "content": "First question"}])

    second_id = store.new_id()
    store.save(second_id, [{"role": "user", "content": "Second question"}])

    conversations = store.list_conversations()

    assert [c["id"] for c in conversations] == [second_id, first_id]


def test_delete_removes_conversation(store):
    conversation_id = store.new_id()
    store.save(conversation_id, _messages())
    assert len(store.list_conversations()) == 1

    store.delete(conversation_id)

    assert store.list_conversations() == []
    assert store.load(conversation_id) == []


def test_delete_nonexistent_conversation_does_not_raise(store):
    store.delete("does-not-exist")  # should not raise


def test_saving_again_updates_the_same_conversation(store):
    conversation_id = store.new_id()
    store.save(conversation_id, _messages())
    store.save(conversation_id, _messages() + [{"role": "user", "content": "Follow-up"}])

    assert len(store.list_conversations()) == 1
    loaded = store.load(conversation_id)
    assert len(loaded) == 3
