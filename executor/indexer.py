import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
from bidict import bidict
from bloom_filter2 import BloomFilter
from jina import Document, DocumentArray, Executor, requests
from docarray.score import NamedScore
from jina.logging.logger import JinaLogger

from .storage import StorageFactory


class FaissIndexer(Executor):
    """A vector similarity indexer for very large scale data using Faiss.

    The documents can be stored using different storage backend (e.g., LMDB, SQLite, PostgresSQL),
    while the vector embeddings are indexed in a FAISS Index.
    """

    def __init__(
        self,
        storage_backend: str = 'lmdb',
        index_key: str = 'Flat',
        metric: str = 'cosine',
        trained_index_file: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        :param storage_backend: the storage backend, e.g., ``LMDB``, ``postgresql+psycopg2://``
        :param index_key: create a new FAISS index of the specified type.
                The type is determined from the given string following the conventions
                of the original FAISS index factory.
                    Recommended options:
                    - "Flat" (default): Best accuracy (= exact). Becomes slow and RAM intense for > 1 Mio docs.
                    - "HNSW": Graph-based heuristic. If not further specified,
                        we use a RAM intense, but more accurate config:
                        HNSW256, efConstruction=256 and efSearch=256
                    - "IVFx,Flat": Inverted Index. Replace x with the number of centroids aka nlist.
                        Rule of thumb: nlist = 10 * sqrt (num_docs) is a good starting point.
                    For more details see:
                    - Overview of indices https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
                    - Guideline for choosing an index https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
                    - FAISS Index factory https://github.com/facebookresearch/faiss/wiki/The-index-factory
        :param trained_index_file: the index file dumped from a trained index, e.g., ``faiss.index``.
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        self.index_key = index_key
        self.metric = metric

        workspace = Path(self.workspace)
        storage_path = str(workspace / 'lmdb_docs')

        self._metas = {'doc_id_to_offset': bidict(), 'delete_marks': []}

        self._bloom = self._init_bloom()

        # the kv_db is the storage backend for documents
        self.logger.info(f'Using "{storage_backend}" as the storage backend')
        self._kv_db = StorageFactory.open(storage_backend, db_path=storage_path)

        # the buffer_indexer is created for incremental updates
        self._buffer_indexer = DocumentArray()

        # the vec_indexer is created for incremental indexing
        self._vec_indexer = None
        if trained_index_file:
            if os.path.exists(trained_index_file):
                self._vec_indexer = faiss.read_index(trained_index_file)
            else:
                raise ValueError(
                    f'The trained index file {trained_index_file} does not exist!'
                )

        self._index_kwargs = kwargs
        self._vec_indexer = self._build_indexer(self._vec_indexer, **kwargs)

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Optional[Dict] = None, **kwargs):
        """Add docs to the index
        :param docs: the documents to add
        :param parameters: parameters to the request
        """

        if (docs is None) or len(docs) == 0:
            return

        sync = parameters.get('sync', True) if parameters else True

        self._kv_db.put(docs)

        if sync:
            new_docs, exist_docs = self.bloom_filter(docs)
            if len(new_docs) > 0:
                doc_ids = new_docs[:, 'id']
                embeddings = new_docs.embeddings
                self._vec_indexer = self._add_vecs_with_ids(
                    self._vec_indexer, embeddings, doc_ids
                )
            if len(exist_docs) > 0:
                self.update(docs, parameters=parameters)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Optional[Dict] = None, **kwargs):
        top_k = int(parameters.get('top_k', 10)) if parameters else 10
        if (docs is None) or len(docs) == 0:
            return

        if self._vec_indexer is None:
            self.logger.warning(f'The indexer has not been initialized!')
            return

        if not self._vec_indexer.is_trained:
            self.logger.warning(f'The indexer need to be trained!')
            return

        embeddings = docs.embeddings.astype(np.float32)

        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)

        if self.total_deletes > 100:
            self.logger.warning(
                f'There are {self.total_deletes} (> 100) documents are deleted, '
                'which will degrade the search performance'
            )

        expand_top_k = 2 * top_k + self.total_deletes

        dists, ids = self._vec_indexer.search(embeddings, expand_top_k)
        if self.metric == 'cosine':
            dists = 1.0 - dists

        if len(self._buffer_indexer) > 0:
            match_args = {'limit': top_k, 'metric': self.metric}
            docs.match(self._buffer_indexer, **match_args)

        for doc_idx, matches in enumerate(zip(ids, dists)):
            buffer_matched_docs = deepcopy(docs[doc_idx].matches)
            matched_docs = OrderedDict()
            for m_info in zip(*matches):
                idx, dist = m_info
                if idx < 0 or self._metas['delete_marks'][idx]:
                    continue
                match_doc_id = self._metas['doc_id_to_offset'].inverse[idx]
                match = self.get_doc(match_doc_id)
                s = NamedScore()
                s.value = dist
                s.ref_id = docs[0].id
                match.scores[self.metric] = s
                matched_docs[match.id] = match

            # merge search results
            for m in buffer_matched_docs:
                matched_docs[m.id] = m
            docs[doc_idx].matches = [
                m
                for _, m in sorted(
                    matched_docs.items(),
                    key=lambda item: item[1].scores[self.metric].value,
                )
            ][:top_k]


    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Optional[Dict] = None, **kwargs):
        """Update entries from the index by id.
        :param docs: the documents to update
        :param parameters: parameters to the request
        """


        if (docs is None) or len(docs) == 0:
            return

        self._kv_db.update(docs)
        sync = parameters.get('sync', True) if parameters else True
        if sync:
            self._buffer_indexer.extend(docs)

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id
        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])
        self._kv_db.delete(deleted_ids)

        # delete from buffer_indexer
        for doc_id in deleted_ids:
            idx = self._metas['doc_id_to_offset'][doc_id]
            self._metas['delete_marks'][idx] = 1

            if doc_id in self._buffer_indexer:
                del self._buffer_indexer[doc_id]

    @requests(on='/sync')
    def sync(self, **kwargs):
        """Sync the data from the storage into the Faiss index"""
        if self._vec_indexer:
            self._vec_indexer.reset()
        self._bloom = self._init_bloom()
        self._metas = {'doc_id_to_offset': bidict(), 'delete_marks': []}
        self._vec_indexer = self._build_indexer(self._vec_indexer)
        self._buffer_indexer.clear()

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Completely clear the storage and index."""
        self._kv_db.clear()
        if self._vec_indexer:
            self._vec_indexer.reset()
        self._buffer_indexer.clear()
        self._metas = {'doc_id_to_offset': bidict(), 'delete_marks': []}
        self._bloom = self._init_bloom()

    @requests(on='/status')
    def status(self, **kwargs):
        """Return the status of the indexer."""
        status = Document(tags={'db_stat': self._kv_db.stat})
        status.tags['total_indexes'] = self.total_indexes
        status.tags['total_updates'] = self.total_updates
        status.tags['total_deletes'] = self.total_deletes
        return status

    def get_doc(self, doc_id: str):
        docs = self._kv_db.get(doc_id)
        if len(docs) == 0:
            return None
        return docs[0]

    def _init_bloom(self):
        # max_elements (100M) is how many elements you expect the filter to hold.
        # error_rate defines accuracy;
        return BloomFilter(max_elements=100000000, error_rate=0.01)

    def _init_indexer(
        self,
        num_dim: int,
        index_key: str = 'Flat',
        metric_type=faiss.METRIC_INNER_PRODUCT,
        **kwargs,
    ):
        if index_key.endswith('HNSW') and metric_type == faiss.METRIC_INNER_PRODUCT:
            # faiss index factory doesn't give the same results for HNSW IP, therefore direct init.
            n_links = kwargs.get('n_links', 128)
            indexer = faiss.IndexHNSWFlat(num_dim, n_links, metric_type)
            indexer.hnsw.efSearch = kwargs.get('efSearch', 20)  # 20
            indexer.hnsw.efConstruction = kwargs.get('efConstruction', 80)  # 80
            self.logger.info(
                f'HNSW params: n_links: {n_links}, efSearch: {indexer.hnsw.efSearch},'
                f' efConstruction: {indexer.hnsw.efConstruction}'
            )
        else:
            indexer = faiss.index_factory(num_dim, index_key, metric_type)

        # # Set verbosity level
        # indexer.verbose = self.verbose
        # if hasattr(indexer, "index") and indexer.index is not None:
        #     indexer.verbose = self.verbose
        # if hasattr(indexer, "quantizer") and indexer.quantizer is not None:
        #     indexer.quantizer.verbose = self.verbose
        # if hasattr(indexer, "clustering_index") and indexer.clustering_index is not None:
        #     indexer.clustering_index.verbose = self.verbose
        return indexer

    def _build_indexer(self, indexer, **kwargs):
        for docs in self._kv_db.batched_iterator():
            doc_ids = docs[:,'id']
            embeddings = docs.embeddings
            N, D = embeddings.shape
            assert len(doc_ids) == N

            indexer = self._add_vecs_with_ids(indexer, embeddings, doc_ids)
            self.append_bloom(docs)
        return indexer

    def _add_vecs_with_ids(
        self, indexer, embeddings: Union[np.ndarray, List], doc_ids: List[str]
    ):
        num_docs = len(doc_ids)
        assert num_docs == len(embeddings)

        if num_docs == 0:
            return

        if isinstance(embeddings, list):
            embeddings = np.stack(embeddings)

        if indexer is None:
            indexer = self._init_indexer(
                embeddings.shape[1],
                index_key=self.index_key,
                metric_type=self.metric_type,
                **self._index_kwargs,
            )

        if not indexer.is_trained:
            self.logger.warning(
                'The new documents will not be indexed, as the indexer need to been'
                ' trained'
            )
            return indexer

        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)

        total_indexes = indexer.ntotal
        for idx, doc_id in zip(range(total_indexes, total_indexes + num_docs), doc_ids):
            self._metas['delete_marks'].append(0)
            self._metas['doc_id_to_offset'][doc_id] = idx

        indexer.add(embeddings)
        return indexer

    def bloom_filter(self, docs: DocumentArray):
        new_docs = DocumentArray()
        exist_docs = DocumentArray()
        for doc in docs:
            if doc.id in self._bloom:
                exist_docs.append(doc)
            else:
                new_docs.append(doc)
        return new_docs, exist_docs

    def append_bloom(self, docs: DocumentArray):
        for doc in docs:
            self._bloom.add(doc.id)

    @property
    def num_dim(self):
        if self._vec_indexer:
            return self._vec_indexer.d
        return None

    @property
    def total_indexes(self):
        return self._vec_indexer.ntotal if self._vec_indexer else 0

    @property
    def size(self):
        return self.total_indexes - self.total_deletes

    @property
    def total_deletes(self):
        return sum(self._metas['delete_marks'])

    @property
    def total_updates(self):
        return len(self._buffer_indexer)

    @property
    def metric_type(self):
        metric_type = faiss.METRIC_L2
        if self.metric == 'cosine':
            self.logger.warning(
                'cosine distance will be output as normalized inner_product distance.'
            )
            metric_type = faiss.METRIC_INNER_PRODUCT

        if self.metric not in {'euclidean', 'cosine'}:
            self.logger.warning(
                'Invalid distance metric for Faiss index construction. Defaulting '
                'to euclidean distance'
            )
        return metric_type
