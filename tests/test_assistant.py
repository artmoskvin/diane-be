import pytest
from langchain.schema import HumanMessage, AIMessage

from diane_be.assistant import IMPROVEMENT_PROMPT, NotesAssistant, NotesAssistantException

REQUEST = "test-request"
RESPONSE = "Test Answer"
NOTE = "Test Note"
TRANSCRIPT = "Test Transcript"
USER_ID = "test-user"


class TestNotesAssistant:
    @pytest.fixture
    def mock_chat_llm(self, mocker):
        return mocker.patch('diane_be.assistant.BaseChatModel')

    @pytest.fixture
    def mock_storage(self, mocker):
        return mocker.patch('diane_be.assistant.NoteStorage')

    @pytest.fixture
    def assistant(self, mock_chat_llm, mock_storage):
        return NotesAssistant(mock_chat_llm, mock_storage)

    def test_add_note_from_transcript_success(self, assistant, mock_chat_llm, mock_storage):
        mock_chat_llm.return_value = AIMessage(content=NOTE)
        mock_storage.insert.return_value = None

        assistant = NotesAssistant(mock_chat_llm, mock_storage)
        result = assistant.add_note_from_transcript(USER_ID, TRANSCRIPT)

        assert result == NOTE

        mock_chat_llm.assert_called_once_with(
            [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript=TRANSCRIPT))])

        mock_storage.insert.assert_called_once_with(USER_ID, NOTE)

    def test_add_note_from_transcript_failure(self, assistant, mock_chat_llm, mock_storage):
        mock_chat_llm.side_effect = Exception("Chat LLM failure")

        with pytest.raises(NotesAssistantException) as exc_info:
            assistant.add_note_from_transcript(USER_ID, TRANSCRIPT)

        assert "Note improvement failed" in str(exc_info.value)
        mock_chat_llm.assert_called_once_with(
            [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript=TRANSCRIPT))])

    def test_insert_index_failure(self, assistant, mock_chat_llm, mock_storage):
        mock_chat_llm.return_value = AIMessage(content=NOTE)
        mock_storage.insert.side_effect = Exception("Index insert failure")

        with pytest.raises(NotesAssistantException) as exc_info:
            assistant.add_note_from_transcript(USER_ID, TRANSCRIPT)

        assert "Note improvement failed" in str(exc_info.value)
        mock_chat_llm.assert_called_once_with(
            [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript=TRANSCRIPT))])
        mock_storage.insert.assert_called_once()

    def test_answer_question_success(self, assistant, mock_chat_llm, mock_storage):
        mock_storage.query.return_value = RESPONSE

        result = assistant.answer_question(USER_ID, REQUEST)

        assert result == RESPONSE

        mock_chat_llm.assert_not_called()

    def test_answer_question_empty_response(self, assistant, mock_chat_llm, mock_storage):
        mock_storage.query.return_value = None
        mock_chat_llm.return_value = AIMessage(content=RESPONSE)

        result = assistant.answer_question(USER_ID, REQUEST)

        assert result == RESPONSE

        mock_chat_llm.assert_called_once_with([HumanMessage(content=REQUEST)])

    def test_query_engine_failure(self, assistant, mock_chat_llm, mock_storage):
        mock_storage.query.side_effect = Exception("Query engine failure")

        with pytest.raises(NotesAssistantException) as exc_info:
            assistant.answer_question(USER_ID, REQUEST)

        assert "Failed answering question" in str(exc_info.value)
        mock_chat_llm.assert_not_called()

    def test_answer_question_empty_response_llm_failure(self, assistant, mock_chat_llm, mock_storage):
        mock_storage.query.return_value = None
        mock_chat_llm.side_effect = Exception("Chat LLM failure")

        with pytest.raises(NotesAssistantException) as exc_info:
            assistant.answer_question(USER_ID, REQUEST)

        assert "Failed answering question" in str(exc_info.value)
        mock_chat_llm.assert_called_once_with([HumanMessage(content=REQUEST)])
