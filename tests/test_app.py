from unittest.mock import create_autospec

from flask import current_app

from diane_be import NotesAssistant

TEST_RESPONSE = "Test response"
TEST_QUESTION = "Test question"
TEST_TRANSCRIPT = "Test transcript"
TEST_NOTE = "Test note"
USER_ID = "test-user-id"


class TestApp:
    def test_add_note_succeeds(self, app, client):
        mock_notes_assistant = create_autospec(NotesAssistant)
        mock_notes_assistant.add_note_from_transcript.return_value = TEST_NOTE

        with app.app_context():
            current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

        response = client.post(
            "/notes",
            json={"transcript": TEST_TRANSCRIPT},
            headers={"user_id": USER_ID})

        assert response.status_code == 200
        assert response.json["note"] == TEST_NOTE

    def test_add_note_fails(self, app, client):
        mock_notes_assistant = create_autospec(NotesAssistant)
        mock_notes_assistant.add_note_from_transcript.side_effect = Exception("Test")

        with app.app_context():
            current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

        response = client.post(
            "/notes",
            json={"transcript": TEST_TRANSCRIPT},
            headers={"user_id": USER_ID})

        assert response.status_code == 500
        assert "Failed adding a note" in response.text

    def test_add_note_transcript_is_empty(self, client):
        response = client.post("/notes", json={}, headers={"user_id": USER_ID})
        assert response.status_code == 400
        assert "Empty payload" in response.text

    def test_add_note_without_user_id(self, client):
        response = client.post("/notes", json={"transcript": TEST_TRANSCRIPT})
        assert response.status_code == 400
        assert "Empty user_id" in response.text

    def test_answer_question_succeeds(self, app, client):
        mock_notes_assistant = create_autospec(NotesAssistant)
        mock_notes_assistant.answer_question.return_value = TEST_RESPONSE

        with app.app_context():
            current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

        response = client.get(
            "/notes/ask",
            query_string={"q": TEST_QUESTION},
            headers={"user_id": USER_ID})

        assert response.status_code == 200
        assert response.json["response"] == "Test response"

    def test_answer_question_fails(self, app, client):
        mock_notes_assistant = create_autospec(NotesAssistant)
        mock_notes_assistant.answer_question.side_effect = Exception("Test")

        with app.app_context():
            current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

        response = client.get(
            "/notes/ask",
            query_string={"q": TEST_QUESTION},
            headers={"user_id": USER_ID})

        assert response.status_code == 500
        assert "Failed answering the question" in response.text

    def test_answer_question_question_is_empty(self, client):
        response = client.get("/notes/ask", headers={"user_id": USER_ID})
        assert response.status_code == 400
        assert "Empty payload" in response.text

    def test_answer_question_userid_is_empty(self, client):
        response = client.get("/notes/ask", query_string={"q": TEST_QUESTION})
        assert response.status_code == 400
        assert "Empty user_id" in response.text
