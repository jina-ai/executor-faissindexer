import pytest
from jina import Document, DocumentArray, Executor, Flow
import numpy as np
from executor.storage import StorageFactory


@pytest.mark.parametrize('backend', ['lmdb', 'sqlite:///:memory:'])
def test_factory(tmpdir, backend):
    storage = StorageFactory.open(backend, db_path=str(tmpdir / 'lmdb.db'))


@pytest.mark.parametrize('backend', ['lmdb', 'sqlite:///:memory:'])
def test_get(tmpdir, backend, docs):
    storage = StorageFactory.open(backend, db_path=str(tmpdir / 'lmdb.db'))
    storage.put(docs)

    doc = storage.get('doc1')[0]
    assert doc.id == 'doc1'
    assert (doc.embedding == [1, 0, 0, 0]).all()

    docs = storage.get('doc7')
    assert len(docs) == 0


@pytest.mark.parametrize('backend', ['lmdb', 'sqlite:///:memory:'])
def test_update(tmpdir, backend, docs, update_docs):
    storage = StorageFactory.open(backend, db_path=str(tmpdir / 'lmdb.db'))
    storage.put(docs)

    storage.update(update_docs)

    doc = storage.get('doc1')[0]
    assert (doc.embedding == [0, 0, 0, 1]).all()


@pytest.mark.parametrize('backend', ['lmdb', 'sqlite:///:memory:'])
def test_delete(tmpdir, backend, docs):
    storage = StorageFactory.open(backend, db_path=str(tmpdir / 'lmdb.db'))
    storage.put(docs)
    storage.delete(['doc1'])
    docs = storage.get('doc1')
    assert len(docs) == 0


@pytest.mark.parametrize('backend', ['lmdb', 'sqlite:///:memory:'])
def test_clear(tmpdir, backend, docs):
    storage = StorageFactory.open(backend, db_path=str(tmpdir / 'lmdb.db'))
    storage.put(docs)

    assert storage.size == 6
    storage.clear()
    assert storage.size == 0


@pytest.mark.parametrize('backend', ['lmdb', 'sqlite:///:memory:'])
def test_batched_iterator(tmpdir, backend, docs):
    storage = StorageFactory.open(backend, db_path=str(tmpdir / 'lmdb.db'))
    storage.put(docs)
    for docs in storage.batched_iterator(batch_size=3):
        assert len(docs) == 3
