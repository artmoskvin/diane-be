import pytest
from unittest.mock import MagicMock, patch, create_autospec

from llama_index import VectorStoreIndex, Response
from llama_index.indices.base import BaseIndex
from llama_index.indices.query.base import BaseQueryEngine

from diane_be.storage import BASE_PATH, NoteStorage, NoteStorageException

RESPONSE = "test-response"
REQUEST = "test-request"
NOTE = "test-note"
USER_ID = "test-user"


class TestNoteStorage:

    def test_insert_succeeds_when_index_exists_in_cache(self):
        mock_s3 = MagicMock()
        mock_index = MagicMock()
        storage = NoteStorage(mock_s3)
        storage.indices = {USER_ID: mock_index}
        storage.insert(USER_ID, NOTE)
        mock_index.insert.assert_called_once()
        mock_index.storage_context.persist.assert_called_once_with(persist_dir=BASE_PATH, fs=mock_s3)

    def test_insert_succeeds_when_index_exists_in_storage(self):
        mock_s3 = MagicMock()
        mock_index = MagicMock()
        storage = NoteStorage(mock_s3)
        with patch.object(storage, '_load_index', return_value=mock_index):
            storage.insert(USER_ID, NOTE)
            mock_index.insert.assert_called_once()
            mock_index.storage_context.persist.assert_called_once_with(persist_dir=BASE_PATH, fs=mock_s3)
            assert USER_ID in storage.indices

    @patch('diane_be.storage.VectorStoreIndex')
    def test_insert_succeeds_when_index_does_not_exist(self, mock_index_class):
        mock_index = MagicMock(spec=VectorStoreIndex)
        mock_index_class.return_value = mock_index
        mock_s3 = MagicMock()
        storage = NoteStorage(mock_s3)
        with patch.object(storage, '_load_index', side_effect=ValueError):
            storage.insert(USER_ID, NOTE)
            mock_index.set_index_id.called_once_with(USER_ID)
            mock_index.insert.assert_called_once()
            mock_index.storage_context.persist.assert_called_once_with(persist_dir=BASE_PATH, fs=mock_s3)
            assert USER_ID in storage.indices

    def test_insert_fails_when_loading_index_fails(self):
        mock_s3 = MagicMock()
        storage = NoteStorage(mock_s3)
        with pytest.raises(NoteStorageException):
            with patch.object(storage, '_load_index', side_effect=Exception):
                storage.insert(USER_ID, NOTE)

    def test_query_when_index_exists_in_cache(self):
        mock_s3 = MagicMock()
        storage = NoteStorage(mock_s3)

        mock_index = create_autospec(spec=BaseIndex)
        mock_query_engine = create_autospec(spec=BaseQueryEngine)
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = Response(response=RESPONSE)
        storage.indices = {USER_ID: mock_index}

        response = storage.query(USER_ID, REQUEST)

        assert response == RESPONSE

    def test_query_when_index_exists_in_storage(self):
        mock_s3 = MagicMock()
        storage = NoteStorage(mock_s3)

        mock_index = create_autospec(spec=BaseIndex)
        mock_query_engine = create_autospec(spec=BaseQueryEngine)
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = Response(response=RESPONSE)

        with patch.object(storage, '_load_index', return_value=mock_index):
            response = storage.query(USER_ID, REQUEST)

            assert response == RESPONSE
            assert USER_ID in storage.indices

    @patch('diane_be.storage.VectorStoreIndex')
    def test_query_when_index_does_not_exist(self, mock_index_class):
        mock_index = create_autospec(spec=BaseIndex)
        mock_query_engine = create_autospec(spec=BaseQueryEngine)
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = Response(response=RESPONSE)
        mock_index_class.return_value = mock_index

        mock_s3 = MagicMock()
        storage = NoteStorage(mock_s3)
        with patch.object(storage, '_load_index', side_effect=ValueError):
            response = storage.query(USER_ID, REQUEST)

            assert response == RESPONSE
            assert USER_ID in storage.indices
            mock_index.set_index_id.assert_called_once_with(USER_ID)

    def test_query_fails_when_loading_index_fails(self):
        mock_s3 = MagicMock()
        storage = NoteStorage(mock_s3)
        with pytest.raises(NoteStorageException):
            with patch.object(storage, '_load_index', side_effect=Exception):
                storage.query(USER_ID, REQUEST)
