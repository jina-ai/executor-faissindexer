import pytest
from jina import Document, DocumentArray, Executor, Flow
import numpy as np
from executor.storage.sql import SQLStorage


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.array([1, 0, 0, 0])),
            Document(id='doc2', embedding=np.array([0, 1, 0, 0])),
            Document(id='doc3', embedding=np.array([0, 0, 1, 0])),
            Document(id='doc4', embedding=np.array([0, 0, 0, 1])),
            Document(id='doc5', embedding=np.array([1, 0, 1, 0])),
            Document(id='doc6', embedding=np.array([0, 1, 0, 1])),
        ]
    )


@pytest.fixture
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.array([0, 0, 0, 1])),
        ]
    )


@pytest.fixture
def storage(docs):
    storage = SQLStorage(db_url='sqlite:///:memory:')
    storage.put(docs)
    return storage


def test_get(storage):
    doc = storage.get('doc1')
    assert doc.id == 'doc1'
    assert (doc.embedding == [1, 0, 0, 0]).all()

    doc = storage.get('doc7')
    assert doc is None


def test_update(storage, update_docs):
    storage.update(update_docs)

    doc = storage.get('doc1')
    assert (doc.embedding == [0, 0, 0, 1]).all()


def test_delete(storage):
    storage.delete(['doc1'])
    doc = storage.get('doc1')
    assert doc is None


def test_clear(storage):
    assert storage.size == 6
    storage.clear()
    assert storage.size == 0


def test_batched_iterator(storage):
    for docs in storage.batched_iterator(batch_size=3):
        assert len(docs) == 3
