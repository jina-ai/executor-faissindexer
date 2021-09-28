import abc
from jina import Document


class Storage(abc.ABC):
    @abc.abstractmethod
    def open(self):
        ...

    @abc.abstractmethod
    def get(self, doc_id: str) -> Document:
        ...
