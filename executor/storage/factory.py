from .lmdb import LMDBStorage


class StorageFactory:
    @staticmethod
    def open(self, backend: str = 'lmdb', **kwargs):
        if backend == 'lmdb':
            db_path = kwargs['db_path']
            return LMDBStorage(db_path)
        else:
            raise NotImplemented(f'The backend `{backend}` is not supported yet!')
