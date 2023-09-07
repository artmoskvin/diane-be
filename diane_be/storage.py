import logging
from typing import Optional

from llama_index import Document, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.indices.base import BaseIndex

BASE_PATH = 'diane-notes/{user_id}/storage'

logger = logging.getLogger(__name__)


class NoteStorage:
    def __init__(self, fs=None):
        self.fs = fs
        self.indices: dict[str, BaseIndex] = {}

    def insert(self, user_id: str, note: str) -> None:
        index_path = BASE_PATH.format(user_id=user_id)
        index = self._load_or_create_index(index_path)

        index.insert(Document(text=note))
        index.storage_context.persist(persist_dir=index_path, fs=self.fs)

    def query(self, user_id: str, request: str) -> Optional[str]:
        index_path = BASE_PATH.format(user_id=user_id)
        index = self._load_or_create_index(index_path)
        query_engine = index.as_query_engine()
        response = query_engine.query(request)
        return response.response

    def _load_or_create_index(self, path: str) -> BaseIndex:
        """
        Loads or creates user index. If index does not exist in cache, look it up in cloud and put in cache.
        If it doesn't exist in cloud, create new one and put in cache.
        """
        if path in self.indices:
            return self.indices.get(path)

        try:
            index = self._load_index(path)
            self.indices[path] = index
        except (ValueError, FileNotFoundError):
            logger.debug(f"Index not found at {path}. Creating new index.")

            index = VectorStoreIndex([])
            self.indices[path] = index
        except Exception as e:
            raise NoteStorageException(f"Failed getting index from {path}") from e

        return index

    def _load_index(self, path: str) -> BaseIndex:
        sc = StorageContext.from_defaults(persist_dir=path, fs=self.fs)
        return load_index_from_storage(sc)


class NoteStorageException(Exception):
    pass
