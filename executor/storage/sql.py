from typing import List
from jina import Document, DocumentArray
from sqlalchemy import Column, DateTime, String, LargeBinary, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import logging
from .base import Storage

from sqlalchemy.orm import sessionmaker


Base = declarative_base()  # type: Any


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
        """An SQL backed DocumentStore. Currently supports SQLite, PostgreSQL and MySQL backends."""
        self.logger = logging.getLogger(self.__module__.__class__.__name__)
        DocumentModel.__tablename__ = table
        self.engine = create_engine(db_url)
        ORMBase.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def get(self, doc_id: str) -> Document:
        result = (
            self.session.query(DocumentModel)
            .filter(DocumentModel.doc_id == doc_id)
            .all()
        )
        if len(result) == 0:
            return None
        else:
            record = result[0]
            return Document(record.doc_data)

    def put(self, docs: DocumentArray):
        for doc in docs:
            doc.embedding = doc.embedding.astype(np.float32)
            record = DocumentModel(doc_id=doc.id, doc_data=doc.SerializeToString())
            self.session.add(record)
        try:
            self.session.commit()
        except Exception as ex:
            self.logger.error(f'Transaction rollback: {ex.__cause__}')
            self.session.rollback()
            raise ex

    def update(self, docs: DocumentArray):
        for doc in docs:
            doc.embedding = doc.embedding.astype(np.float32)
            record = DocumentModel(doc_id=doc.id, doc_data=doc.SerializeToString())
            self.session.merge(record)
        try:
            self.session.commit()
        except Exception as ex:
            self.logger.error(f'Transaction rollback: {ex.__cause__}')
            self.session.rollback()
            raise ex

    def delete(self, doc_ids: List[str]):
        assert isinstance(doc_ids, list)
        for doc_id in doc_ids:
            self.session.query(DocumentModel).filter(
                DocumentModel.doc_id == doc_id
            ).delete()
        self.session.commit()

    def batched_iterator(self, batch_size: int = 1, **kwargs):
        docs = DocumentArray()
        count = 0
        for record in (
            self.session.query(DocumentModel).yield_per(1).enable_eagerloads(False)
        ):
            docs.append(Document(record.doc_data))
            count += 1
            if count % batch_size == 0:
                yield docs
                docs.clear()
        if len(docs) > 0:
            yield docs

    def clear(self):
        documents = self.session.query(DocumentModel)
        documents.delete(synchronize_session=False)

    @property
    def size(self):
        return self.session.query(func.count(DocumentModel.doc_id)).scalar()

    @property
    def stat(self):
        return {'count': self.size}
