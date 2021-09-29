import shutil
from copy import deepcopy
import pytest
from jina import DocumentArray
from executor import FaissIndexer


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


def test_sync(tmpdir, docs, update_docs):
    metas = {'workspace': str(tmpdir)}

    # index docs first
    indexer = FaissIndexer(metas=metas)
    indexer.index(docs)

    # update first doc
    indexer.update(update_docs)
    indexer.sync()

    assert indexer.total_indexes == 6
    assert indexer.total_deletes == 0
    assert indexer.total_updates == 0
    assert len(indexer._buffer_indexer) == 0
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

    assert indexer.status().tags['db_stat']['entries'] == 0


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


def test_sql_backend(tmpdir, docs, update_docs):
    metas = {'workspace': str(tmpdir)}
    indexer = FaissIndexer(
        storage_backend='sqlite:///:memory:', index_key='Flat', metas=metas
    )
    assert indexer.num_dim is None
    assert indexer.total_indexes == 0
    assert indexer.total_deletes == 0
    assert indexer.total_updates == 0

    indexer.index(docs)
    assert indexer.num_dim == 4
    assert indexer.total_indexes == 6
    assert indexer.total_deletes == 0
    assert indexer.total_updates == 0

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

    # update first doc
    indexer.update(update_docs)
    indexer.sync()

    assert indexer.total_indexes == 6
    assert indexer.total_deletes == 0
    assert indexer.total_updates == 0
    assert len(indexer._buffer_indexer) == 0
    assert (indexer.get_doc(f'doc1').embedding == [0, 0, 0, 1]).all()

    search_docs = deepcopy(update_docs)
    indexer.search(search_docs)
    search_docs[0].matches[0].id == f'doc1'
