import logging

from flask import Blueprint, request, jsonify, abort, current_app

from diane_be.assistant import NotesAssistant

logger = logging.getLogger(__name__)

bp = Blueprint('notes', __name__, url_prefix='/notes')


@bp.route('', methods=['POST'])
def add_note():
    logger.debug("Received add_note request")

    transcript = request.json.get("transcript", None)

    if transcript:
        logger.debug("Transcript: " + transcript)
        try:
            notes_assistant: NotesAssistant = current_app.config['NOTES_ASSISTANT']

            note = notes_assistant.add_note_from_transcript(transcript)

            logger.debug("Note: " + note)

            return jsonify({"note": note})
        except Exception:
            logger.error("Failed adding a note", exc_info=True)
            abort(500, description="Failed adding a note")

    abort(400, description="Empty payload")


@bp.route("/ask", methods=["GET"])
def ask_notes():
    logger.debug("Received ask_notes request")

    question = request.args.get("q", None)

    if question:
        logger.debug("Question: " + question)
        try:
            notes_assistant: NotesAssistant = current_app.config['NOTES_ASSISTANT']

            response = notes_assistant.answer_question(question)

            logger.debug(f"Response: {response}")

            return jsonify({"response": response})
        except Exception:
            logger.error("Failed answering the question", exc_info=True)
            abort(500, description="Failed answering the question")

    abort(400, description="Empty payload")
