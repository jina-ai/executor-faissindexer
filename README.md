# FaissIndexer

A similarity search indexer based on Faiss.

## Usage

- use ``LMDB`` as storage backend (as default)
    ```python
    f = Flow().add(uses='jinahub://FaissIndexer/v0.1', uses_with={'storage_backend': 'lmdb'})
    with f:
        f.block()
    ```

- use ``SQLite`` as storage backend
    ```python
    f = Flow().add(uses='config.yml',
                   uses_with={'storage_backend': 'sqlite:///:memory:'})
    with f:
        f.block()
    ```

- use ``PostgresSQL`` as storage backend
    ```python
    f = Flow().add(uses='jinahub://FaissIndexer/v0.1',
                   uses_with={'storage_backend': 'postgresql+psycopg2://postgres:123456@127.0.0.1/postgres'})
    with f:
        f.block()
    ```
