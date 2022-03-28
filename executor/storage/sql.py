from typing import List, Union
from jina import Document, DocumentArray
from jina.logging.logger import JinaLogger

from sqlalchemy import Column, DateTime, String, LargeBinary, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import numpy as np
from .base import Storage

Base = declarative_base()


class ORMBase(Base):
    __abstract__ = True

    created = Column(DateTime, server_default=func.now())
    updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now())


class DocumentModel(ORMBase):
    __tablename__ = 'document'

    doc_id = Column(String(100), nullable=False, primary_key=True)
    doc_data = Column(LargeBinary, nullable=False)


class SQLStorage(Storage):
    def __init__(self, db_url: str, table: str = 'document', **kwargs):
        """An SQL backed storage. Currently supports SQLite, PostgreSQL and MySQL backends."""

        DocumentModel.__tablename__ = table
        self.engine = create_engine(db_url)
        ORMBase.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

        self.logger = JinaLogger(self.__class__.__name__)

    def get(self, doc_ids: Union[str, list]) -> DocumentArray:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        docs = DocumentArray()
        for record in (
            self.session.query(DocumentModel)
            .filter(DocumentModel.doc_id.in_(doc_ids))
            .all()
        ):
            doc = Document.from_bytes(record.doc_data)
            docs.append(doc)
        return docs

    def put(self, docs: DocumentArray):
        for doc in docs:
            doc.embedding = doc.embedding.astype(np.float32)
            record = DocumentModel(doc_id=doc.id, doc_data=doc.to_bytes())
            self.session.merge(record)
        try:
            self.session.commit()
        except Exception as ex:
            self.logger.error(f'Transaction rollback: {ex.__cause__}')
            self.session.rollback()
            raise ex
        self.logger.debug(f'Add {len(docs)} documents')

    def update(self, docs: DocumentArray):
        for doc in docs:
            doc.embedding = doc.embedding.astype(np.float32)
            record = DocumentModel(doc_id=doc.id, doc_data=doc.to_bytes())
            self.session.merge(record)
        try:
            self.session.commit()
        except Exception as ex:
            self.logger.error(f'Transaction rollback: {ex.__cause__}')
            self.session.rollback()
            raise ex
        self.logger.debug(f'Update {len(docs)} documents')

    def delete(self, doc_ids: List[str]):
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]

        for doc_id in doc_ids:
            self.session.query(DocumentModel).filter(
                DocumentModel.doc_id == doc_id
            ).delete()
        self.session.commit()
        self.logger.debug(f'Delete {len(doc_ids)} documents')

    def batched_iterator(self, batch_size: int = 1, **kwargs):
        docs = DocumentArray()
        count = 0
        for record in (
            self.session.query(DocumentModel).yield_per(1).enable_eagerloads(False)
        ):
            docs.append(Document.from_bytes(record.doc_data))
            count += 1
            if count % batch_size == 0:
                yield docs
                docs.clear()
        if len(docs) > 0:
            yield docs

    def clear(self):
        documents = self.session.query(DocumentModel)
        documents.delete(synchronize_session=False)
        self.logger.info(f'Clear documents storage')

    @property
    def size(self):
        return self.session.query(func.count(DocumentModel.doc_id)).scalar()

    @property
    def stat(self):
        return {'count': self.size}
