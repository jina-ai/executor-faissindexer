import abc
from typing import List
from jina import Document, DocumentArray


class Storage(abc.ABC):
    @abc.abstractmethod
    def get(self, doc_id: str) -> Document:
        ...

    @abc.abstractmethod
    def put(self, docs: DocumentArray):
        pass

    @abc.abstractmethod
    def update(self, docs: DocumentArray):
        pass

    @abc.abstractmethod
    def delete(self, doc_ids: List[str]):
        pass

    @abc.abstractmethod
    def clear(self):
        pass
