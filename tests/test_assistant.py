from unittest.mock import ANY
from langchain.schema import HumanMessage, AIMessage
from llama_index import Document, Response
import pytest

from diane_be.assistant import IMPROVEMENT_PROMPT, NotesAssistant, NotesAssistantException


class TestNotesAssistant:
    @pytest.fixture
    def mock_chat_llm(self, mocker):
        return mocker.patch('diane_be.assistant.BaseChatModel')

    @pytest.fixture
    def mock_index(self, mocker):
        return mocker.patch('diane_be.assistant.BaseIndex')

    @pytest.fixture
    def assistant(self, mock_chat_llm, mock_index):
        return NotesAssistant(mock_chat_llm, mock_index)

    def test_add_note_from_transcript_success(self, assistant, mock_chat_llm, mock_index):
        mock_chat_llm.return_value = AIMessage(content="Test Note")
        mock_index.insert.return_value = None
        mock_index.storage_context.persist.return_value = None

        assistant = NotesAssistant(mock_chat_llm, mock_index)
        result = assistant.add_note_from_transcript("Test Transcript")

        assert result == "Test Note"

        mock_chat_llm.assert_called_once_with(
            [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript="Test Transcript"))])

        expected_document = Document(text="Test Note")
        expected_document.id_ = ANY

        mock_index.insert.assert_called_once_with(expected_document)
        mock_index.storage_context.persist.assert_called_once()

    def test_add_note_from_transcript_failure(self, assistant, mock_chat_llm, mock_index):
        mock_chat_llm.side_effect = Exception("Chat LLM failure")
        mock_index.insert.return_value = None
        mock_index.storage_context.persist.return_value = None

        assistant = NotesAssistant(mock_chat_llm, mock_index)
        with pytest.raises(NotesAssistantException) as exc_info:
            assistant.add_note_from_transcript("Test Transcript")

        assert "Note improvement failed" in str(exc_info.value)
        mock_chat_llm.assert_called_once_with(
            [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript="Test Transcript"))])
        mock_index.insert.assert_not_called()
        mock_index.storage_context.persist.assert_not_called()

    def test_insert_index_failure(self, assistant, mock_chat_llm, mock_index):
        mock_chat_llm.return_value = AIMessage(content="Test Note")
        mock_index.insert.side_effect = Exception("Index insert failure")
        mock_index.storage_context.persist.return_value = None

        assistant = NotesAssistant(mock_chat_llm, mock_index)
        with pytest.raises(NotesAssistantException) as exc_info:
            assistant.add_note_from_transcript("Test Transcript")

        assert "Note improvement failed" in str(exc_info.value)
        mock_chat_llm.assert_called_once_with(
            [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript="Test Transcript"))])
        mock_index.insert.assert_called_once()
        mock_index.storage_context.persist.assert_not_called()

    def test_persist_storage_context_failure(self, assistant, mock_chat_llm, mock_index):
        mock_chat_llm.return_value = AIMessage(content="Test Note")
        mock_index.insert.return_value = None
        mock_index.storage_context.persist.side_effect = Exception("Persist storage context failure")

        assistant = NotesAssistant(mock_chat_llm, mock_index)
        with pytest.raises(NotesAssistantException) as exc_info:
            assistant.add_note_from_transcript("Test Transcript")

        assert "Note improvement failed" in str(exc_info.value)
        mock_chat_llm.assert_called_once_with(
            [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript="Test Transcript"))])
        mock_index.insert.assert_called_once_with(ANY)
        mock_index.storage_context.persist.assert_called_once()

    def test_answer_question_success(self, assistant, mock_chat_llm, mock_index):
        mock_index.as_query_engine.return_value.query.return_value = Response(response="Test Answer")

        result = assistant.answer_question("Test Question")

        assert result == "Test Answer"

        mock_chat_llm.assert_not_called()

    def test_answer_question_empty_response(self, assistant, mock_chat_llm, mock_index):
        mock_index.as_query_engine.return_value.query.return_value = Response(response=None)
        mock_chat_llm.return_value = AIMessage(content="Fallback Answer")

        result = assistant.answer_question("Test Question")

        assert result == "Fallback Answer"

        mock_chat_llm.assert_called_once_with([HumanMessage(content="Test Question")])

    def test_query_engine_failure(self, assistant, mock_chat_llm, mock_index):
        mock_index.as_query_engine.return_value.query.side_effect = Exception("Query engine failure")

        with pytest.raises(NotesAssistantException) as exc_info:
            result = assistant.answer_question("Test Question")

        assert "Failed answering question" in str(exc_info.value)
        mock_chat_llm.assert_not_called()

    def test_answer_question_empty_response_llm_failure(self, assistant, mock_chat_llm, mock_index):
        mock_index.as_query_engine.return_value.query.return_value = Response(response=None)
        mock_chat_llm.side_effect = Exception("Chat LLM failure")

        with pytest.raises(NotesAssistantException) as exc_info:
            result = assistant.answer_question("Test Question")

        assert "Failed answering question" in str(exc_info.value)
        mock_chat_llm.assert_called_once_with([HumanMessage(content="Test Question")])
