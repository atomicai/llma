import abc
from typing import Optional, Dict, List
from llma.etc.schema import Document
from llma.etc.errors import DuplicateDocumentError
from llma.etc.filters import FilterType
import numpy as np
from loguru import logger


class INNDocStore(abc.ABC):

    index: Optional[str]
    similarity: Optional[str]
    duplicate_documents_options: tuple = ("skip", "overwrite", "fail")
    ids_iterator = None

    @abc.abstractmethod
    def write_documents(self, docs, index: str = None):
        pass

    @abc.abstractmethod
    def get_all_documents(self):
        pass

    @abc.abstractmethod
    def delete_all_documents(self, index: Optional[str] = None):
        pass

    @abc.abstractclassmethod
    def get_document_by_id(
        self,
        id: str,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Document]:
        pass

    @abc.abstractmethod
    def count_documents(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        pass

    def __iter__(self):
        if not self.ids_iterator:
            self.ids_iterator = [x.id for x in self.get_all_documents()]
        return self

    def describe_documents(self, index=None):
        """
        Statistics of the documents
        """
        if index is None:
            index = self.index
        docs = self.get_all_documents(index)
        lens = [len(d.content) for d in docs]
        response = dict(
            count=len(docs),
            chars_mean=np.mean(lens),
            chars_max=max(lens),
            chars_min=min(lens),
            chars_median=np.median(lens),
        )
        return response

    def __next__(self):
        if len(self.ids_iterator) == 0:
            raise StopIteration
        curr_id = self.ids_iterator[0]
        ret = self.get_document_by_id(curr_id)
        self.ids_iterator = self.ids_iterator[1:]
        return ret

    def _drop_duplicate_documents(
        self, documents: List[Document], index: Optional[str] = None
    ) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID
        :param documents: A list of Document objects.
        :param index: name of the index
        :return: A list of Document objects.
        """
        _hash_ids = set([])
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already exists in index '%s'",
                    document.id,
                    index or self.index,
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents

    def _handle_duplicate_documents(
        self,
        documents: List[Document],
        index: Optional[str] = None,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.
        :param documents: A list of Document objects.
        :param index: name of the index
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :return: A list of Document objects.
        """

        index = index or self.index
        if duplicate_documents in ("skip", "fail"):
            documents = self._drop_duplicate_documents(documents, index)
            documents_found = self.get_documents_by_id(
                ids=[doc.id for doc in documents], index=index, headers=headers
            )
            ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and duplicate_documents == "fail":
                raise DuplicateDocumentError(
                    f"Document with ids '{', '.join(ids_exist_in_db)} already exists in index = '{index}'."
                )

            documents = list(
                filter(lambda doc: doc.id not in ids_exist_in_db, documents)
            )

        return documents

    @abc.abstractmethod
    def search(self, **props):
        pass

    @abc.abstractmethod
    def search_by_embedding(self, **props):
        pass

    @abc.abstractmethod
    def search_by_keywords(self, **props):
        pass
