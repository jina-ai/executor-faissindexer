from typing import Optional
from .lmdb import LMDBStorage
from .sql import SQLStorage


class StorageFactory:
    @staticmethod
    def open(backend: str, **kwargs):
        if backend == 'lmdb':
            db_path = kwargs['db_path']
            kwargs.pop('db_path')
            return LMDBStorage(db_path, **kwargs)
        elif backend:
            return SQLStorage(backend, **kwargs)
        else:
            raise NotImplemented(f'The backend `{backend}` is not supported yet!')
