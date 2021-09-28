import lmdb
from typing import List
from jina import Document, DocumentArray
import numpy as np
from .base import Storage


class LMDBStorage(Storage):
    def __init__(self, path: str):
        self._path = path
        self._env = self.open(path)

    def open(self, db_path: str):
        return lmdb.Environment(
            self._path,
            map_size=int(3.436e10),  # in bytes, 32G,
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

    def put(self, docs: DocumentArray):
        with self._env.begin(write=True) as txn:
            for doc in docs:
                # enforce using float32 as dtype of embeddings
                doc.embedding = doc.embedding.astype(np.float32)
                txn.put(doc.id.encode(), doc.SerializeToString(), overwrite=True)

    def update(self, docs: DocumentArray):
        with self._env.begin(write=True) as txn:
            for doc in docs:
                doc.embedding = doc.embedding.astype(np.float32)
                old_value = txn.replace(doc.id.encode(), doc.SerializeToString())
                if not old_value:
                    txn.abort()
                    raise ValueError(f'The Doc ({doc.id}) does not exist in database!')

    def delete(self, doc_ids: List[str]):
        with self._env.begin(write=True) as txn:
            for doc_id in doc_ids:
                txn.delete(doc_id.encode())

    def get(self, doc_id: str):
        with self._env.begin(write=False) as txn:
            buffer = txn.get(doc_id.encode())
            if buffer:
                return Document(buffer)
            return None

    def clear(self):
        with self._env.begin(write=True) as txn:
            txn.drop(self._env.open_db(txn=txn), delete=False)

    @property
    def stat(self):
        with self._env.begin(write=False) as txn:
            return txn.stat()

    def batched_iterator(self, batch_size: int = 1, **kwargs):
        count = 0
        docs = DocumentArray()
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            cursor.iternext()
            iterator = cursor.iternext(keys=False, values=True)

            for value in iterator:
                doc = Document(value)
                docs.append(doc)
                count += 1
                if count % batch_size == 0:
                    yield docs
                    docs.clear()

            if len(docs) > 0:
                yield docs
