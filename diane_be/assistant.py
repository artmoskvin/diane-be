import logging

from langchain import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage

from diane_be.storage import NoteStorage

IMPROVEMENT_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are a notetaker. Below is an audio transcript recorded from user. Make a proper note from \
this transcript. Don't provide any explanations. Don't wrap the note in quotes.
 
Transcript: {transcript}
Note:""",
)

logger = logging.getLogger(__name__)


class NotesAssistant:
    def __init__(self, chat_llm: BaseChatModel, storage: NoteStorage):
        self.chat_llm = chat_llm
        self.storage = storage

    def add_note_from_transcript(self, user_id: str, transcript: str) -> str:
        messages = [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript=transcript))]

        try:
            note = self.chat_llm(messages).content
            self.storage.insert(user_id, note)
            return note
        except Exception as e:
            raise NotesAssistantException("Note improvement failed") from e

    def answer_question(self, user_id: str, question: str) -> str:
        try:
            response = self.storage.query(user_id, question)
            if not response:
                logger.warning(f"No context found for question '{question}'. Falling back to LLM.")
                return self.chat_llm([HumanMessage(content=question)]).content
            return response
        except Exception as e:
            raise NotesAssistantException("Failed answering question") from e


class NotesAssistantException(Exception):
    pass
