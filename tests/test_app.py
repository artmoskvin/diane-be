from unittest.mock import create_autospec

from flask import current_app

from diane_be import NotesAssistant

TEST_RESPONSE = "Test response"
TEST_QUESTION = "Test question"
TEST_TRANSCRIPT = "Test transcript"
TEST_NOTE = "Test note"


def test_add_note_succeeds(app, client):
    mock_notes_assistant = create_autospec(NotesAssistant)
    mock_notes_assistant.add_note_from_transcript.return_value = TEST_NOTE

    with app.app_context():
        current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

    response = client.post("/notes", json={
        "transcript": TEST_TRANSCRIPT
    })
    assert response.status_code == 200
    assert response.json["note"] == TEST_NOTE
    mock_notes_assistant.add_note_from_transcript.assert_called_once_with(TEST_TRANSCRIPT)


def test_add_note_fails(app, client):
    mock_notes_assistant = create_autospec(NotesAssistant)
    mock_notes_assistant.add_note_from_transcript.side_effect = Exception("Test")

    with app.app_context():
        current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

    response = client.post("/notes", json={
        "transcript": TEST_TRANSCRIPT
    })
    assert response.status_code == 500
    assert "Failed adding a note" in response.text
    mock_notes_assistant.add_note_from_transcript.assert_called_once_with(TEST_TRANSCRIPT)


def test_add_note_transcript_is_empty(client):
    response = client.post("/notes", json={})
    assert response.status_code == 400
    assert "Empty payload" in response.text


def test_answer_question_succeeds(app, client):
    mock_notes_assistant = create_autospec(NotesAssistant)
    mock_notes_assistant.answer_question.return_value = TEST_RESPONSE

    with app.app_context():
        current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

    response = client.get("/notes/ask", query_string={
        "q": TEST_QUESTION
    })
    assert response.status_code == 200
    assert response.json["response"] == "Test response"
    mock_notes_assistant.answer_question.assert_called_once_with(TEST_QUESTION)


def test_answer_question_fails(app, client):
    mock_notes_assistant = create_autospec(NotesAssistant)
    mock_notes_assistant.answer_question.side_effect = Exception("Test")

    with app.app_context():
        current_app.config["NOTES_ASSISTANT"] = mock_notes_assistant

    response = client.get("/notes/ask", query_string={
        "q": TEST_QUESTION
    })
    assert response.status_code == 500
    assert "Failed answering the question" in response.text
    mock_notes_assistant.answer_question.assert_called_once_with(TEST_QUESTION)


def test_answer_question_question_is_empty(client):
    response = client.get("/notes/ask")
    assert response.status_code == 400
    assert "Empty payload" in response.text
