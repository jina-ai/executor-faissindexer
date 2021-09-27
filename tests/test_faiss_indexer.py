import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor, Flow

from executor import FaissIndexer


def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches


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


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[1] / 'config.yml'))
    assert len(ex._buffer_indexer) == 0


@pytest.mark.parametrize('index_key', ['Flat', 'HNSW'])
def test_index(tmpdir, docs, index_key):
    metas = {'workspace': str(tmpdir)}
    indexer1 = FaissIndexer(index_key=index_key, metas=metas)
    assert indexer1.num_dim is None
    assert indexer1.total_indexes == 0
    assert indexer1.total_deletes == 0
    assert indexer1.total_updates == 0

    indexer1.index(docs)
    assert indexer1.num_dim == 4
    assert indexer1.total_indexes == 6
    assert indexer1.total_deletes == 0
    assert indexer1.total_updates == 0

    indexer2 = FaissIndexer(metas=metas)
    assert indexer1.num_dim == 4
    assert indexer1.total_indexes == 6
    assert indexer1.total_deletes == 0
    assert indexer1.total_updates == 0


def test_delete(tmpdir, docs):
    metas = {'workspace': str(tmpdir)}

    # index docs first
    indexer = FaissIndexer(metas=metas)
    indexer.index(docs)

    # delete empty docs
    indexer.delete({})

    # delete first 3 docs
    parameters = {'ids': [f'doc{i}' for i in range(1, 4)]}
    indexer.delete(parameters)
    assert indexer.total_indexes == 6
    assert indexer.total_deletes == 3
    assert indexer.total_updates == 0
    assert indexer.size == 3
    assert indexer.get_doc(f'doc1') is None

    search_docs = deepcopy(docs)
    indexer.search(search_docs)
    search_docs[0].matches[0].id == f'doc5'
    search_docs[1].matches[0].id == f'doc6'
    for i in range(len(docs)):
        assert len(search_docs[i].matches) == 3

    # delete the rest of the docs stored
    parameters = {'ids': [f'doc{i}' for i in range(4, 7)]}
    indexer.delete(parameters)
    assert indexer.total_indexes == 6
    assert indexer.total_deletes == 6
    assert indexer.total_updates == 0
    assert indexer.size == 0
    assert indexer.get_doc(f'doc6') is None

    # delete from empty storage
    parameters = {'ids': [f'doc{i}' for i in range(4, 7)]}
    indexer.delete(parameters)


def test_update(tmpdir, docs, update_docs):
    metas = {'workspace': str(tmpdir)}

    # index docs first
    indexer = FaissIndexer(metas=metas)
    indexer.index(docs)

    # update first doc
    indexer.update(update_docs)
    assert indexer.total_indexes == 6
    assert indexer.total_deletes == 0
    assert indexer.total_updates == 1
    assert (indexer._buffer_indexer[0].embedding == [0, 0, 0, 1]).all()
    assert (indexer.get_doc(f'doc1').embedding == [0, 0, 0, 1]).all()

    search_docs = deepcopy(update_docs)
    indexer.search(search_docs)
    search_docs[0].matches[0].id == f'doc1'


def test_clear(tmpdir, docs, update_docs):
    metas = {'workspace': str(tmpdir)}

    # index docs first
    indexer = FaissIndexer(metas=metas)
    indexer.index(docs)

    indexer.clear()
    assert indexer.total_indexes == 0
    assert indexer.total_deletes == 0
    assert indexer.total_updates == 0

    assert indexer.status().tags['env_stat']['entries'] == 0


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_search(tmpdir, metric, docs):
    metas = {'workspace': str(tmpdir)}

    # test general/normal case
    indexer = FaissIndexer(metric=metric, metas=metas)
    indexer.index(docs)
    search_docs = deepcopy(docs)
    indexer.search(search_docs)

    for i in range(len(docs)):
        assert search_docs[i].matches[0].id == f'doc{i + 1}'
        assert len(search_docs[i].matches) == len(docs)

    # test search with top_k/limit = 1
    indexer.search(search_docs, parameters={'top_k': 1})
    for i in range(len(docs)):
        assert len(search_docs[i].matches) == 1

    # test search from empty indexed docs
    shutil.rmtree(tmpdir)
    indexer = FaissIndexer(metas=metas)
    indexer.index(DocumentArray())
    indexer.search(docs)
    for doc in docs:
        assert not doc.matches

    # test search empty docs
    indexer.search(DocumentArray())
