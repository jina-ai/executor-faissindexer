import logging
from copy import deepcopy

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, List, Union

import faiss
import lmdb
import numpy as np
from jina import Document, Executor, DocumentArray, requests


class FaissIndexer(Executor):
    """A vector similarity indexer based on Faiss and LMDB"""

    def __init__(
        self,
        index_key: str = 'Flat',
        metric: str = 'cosine',
        *args,
        **kwargs,
    ):
        """
        :param index_path: index path
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
        """
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__module__.__class__.__name__)

        self.index_key = index_key
        self.metric = metric

        workspace = Path(self.workspace)
        storage_path = str(workspace / 'lmdb_docs')

        self._metas = {'doc_ids': [], 'doc_id_to_offset': {}, 'delete_marks': []}

        # the kv_storage is the storage backend for documents
        self._kv_storage_env = lmdb.Environment(
            storage_path,
            map_size=3.436e10,  # in bytes, 32G,
            subdir=False,
            readonly=False,
            metasync=True,
            sync=True,
            map_async=False,
            mode=493,
            create=True,
            readahead=True,
            writemap=False,
            meminit=True,
            max_readers=126,
            max_dbs=0,  # means only one db
            max_spare_txns=1,
            lock=True,
        )

        # the buffer_indexer is created for incremental updates
        self._buffer_indexer = DocumentArray()

        # the vec_indexer is created for incremental adding
        self._vec_indexer = None
        self._build_indexer(**kwargs)
        self._index_kwargs = kwargs

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Optional[Dict] = None, **kwargs):
        """Add docs to the index
        :param docs: the documents to add
        :param parameters: parameters to the request
        """

        if docs is None:
            return

        updated_docs = DocumentArray()

        embeddings = []
        doc_ids = []

        with self._kv_storage_env.begin(write=True) as txn:
            for doc in docs:
                # enforce using float32 as dtype of embeddings
                doc.embedding = doc.embedding.astype(np.float32)
                added = txn.put(
                    doc.id.encode(), doc.SerializeToString(), overwrite=True
                )
                if added:
                    embeddings.append(doc.embedding)
                    doc_ids.append(doc.id)
                else:
                    # TODO: use hash to identify fake updates
                    updated_docs.append(doc)

            self._add_vecs_with_ids(embeddings, doc_ids)

        self.update(updated_docs, parameters=parameters)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Optional[Dict] = None, **kwargs):
        top_k = int(parameters.get('top_k', 10)) if parameters else 10
        if (docs is None) or len(docs) == 0:
            return

        if self._vec_indexer is None:
            return

        embeddings = docs.embeddings.astype(np.float32)

        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)

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
                match_doc_id = self._metas['doc_ids'][idx]
                match = self.get_doc(match_doc_id)
                match.scores[self.metric] = dist

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
        """Update entries from the index by id
        :param docs: the documents to update
        :param parameters: parameters to the request
        """

        if docs is None:
            return

        with self._kv_storage_env.begin(write=True) as txn:
            for doc in docs:
                value = txn.replace(doc.id.encode(), doc.SerializeToString())
                if not value:
                    raise ValueError(f'The Doc ({doc.id}) does not exist in database!')
                self._buffer_indexer.append(doc)

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id
        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])

        with self._kv_storage_env.begin(write=True) as txn:
            for doc_id in deleted_ids:
                deleted = txn.delete(doc_id.encode())
                # delete from buffer_indexer
                if deleted:
                    idx = self._metas['doc_id_to_offset'][doc_id]
                    self._metas['delete_marks'][idx] = 1

                    if doc_id in self._buffer_indexer:
                        del self._buffer_indexer[doc_id]
                else:
                    self.logger.warning(
                        f'Can not delete no-existed Doc ({doc_id}) from {self.__module__.__class__.__name__}'
                    )

    def get_doc(self, doc_id: str):
        with self._kv_storage_env.begin(write=False) as txn:
            buffer = txn.get(doc_id.encode())
            if buffer:
                return Document(buffer)
            return None

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
                f'HNSW params: n_links: {n_links}, efSearch: {indexer.hnsw.efSearch}, efConstruction: {indexer.hnsw.efConstruction}'
            )
        else:
            indexer = faiss.index_factory(num_dim, index_key, metric_type)
        return indexer

    def _build_indexer(self, **kwargs):
        with self._kv_storage_env.begin(write=False) as txn:
            cursor = txn.cursor()
            cursor.iternext()
            iterator = cursor.iternext(keys=True, values=True)

            for _id, _data in iterator:
                doc = Document(_data)
                embeds = np.asarray(doc.embedding).reshape((1, -1))

                if self._vec_indexer is None:
                    self._vec_indexer = self._init_indexer(
                        embeds.shape[1], self.index_key, self.metric_type, **kwargs
                    )

                self._add_vecs_with_ids(embeds, [doc.id])

    def _add_vecs_with_ids(
        self, embeddings: Union[np.ndarray, List], doc_ids: List[str]
    ):
        num_docs = len(doc_ids)
        assert num_docs == len(embeddings)

        if num_docs == 0:
            return

        if isinstance(embeddings, list):
            embeddings = np.stack(embeddings)

        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)

        if self._vec_indexer is None:
            self._vec_indexer = self._init_indexer(
                embeddings.shape[1],
                self.index_key,
                self.metric_type,
                **self._index_kwargs,
            )
        total_indexes = self.total_indexes
        for idx, doc_id in zip(range(total_indexes, total_indexes + num_docs), doc_ids):
            self._metas['doc_ids'].append(doc_id)
            self._metas['delete_marks'].append(0)
            self._metas['doc_id_to_offset'][doc_id] = idx

        self._vec_indexer.add(embeddings)

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
        if self.metric == 'inner_product':
            self.logger.warning(
                'inner_product will be output as distance instead of similarity.'
            )
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == 'cosine':
            self.logger.warning(
                'cosine distance will be output as normalized inner_product distance.'
            )
            metric_type = faiss.METRIC_INNER_PRODUCT

        if self.metric not in {'inner_product', 'euclidean', 'cosine'}:
            self.logger.warning(
                'Invalid distance metric for Faiss index construction. Defaulting '
                'to l2 distance'
            )
        return metric_type
