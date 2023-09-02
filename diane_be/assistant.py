import logging

from langchain import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage
from llama_index import Document
from llama_index.indices.base import BaseIndex

IMPROVEMENT_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are a notetaker. Below is an audio transcript recorded from user. Make a proper note from \
this transcript. Don't provide any explanations. Don't wrap the note in quotes.
 
Transcript: {transcript}
Note:""",
)

QUESTION_PROMPT = PromptTemplate(
    input_variables=["question", "relevant_notes"],
    template="""You are a notetaker. You received a question and the list of relevant notes from a user. \
Answer the question using information from the relevant notes. Be concise.
 
Question: {question}

Relevant notes:
''' 
{relevant_notes}
'''

Answer:""",
)

logger = logging.getLogger(__name__)


class NotesAssistant:
    def __init__(self, chat_llm: BaseChatModel, index: BaseIndex):
        self.chat_llm = chat_llm
        self.index = index
        self.query_engine = self.index.as_query_engine()

    def add_note_from_transcript(self, transcript):
        messages = [HumanMessage(content=IMPROVEMENT_PROMPT.format(transcript=transcript))]

        try:
            note = self.chat_llm(messages).content
            self.index.insert(Document(text=note))
            self.index.storage_context.persist()
            return note
        except Exception as e:
            logger.error("Note improvement failed", exc_info=e)
            raise NotesAssistantException("Note improvement failed") from e

    def answer_question(self, question):
        try:
            response = self.query_engine.query(question)
            return response.response
        except Exception as e:
            logger.error("Failed answering question", exc_info=e)
            raise NotesAssistantException("Failed answering question") from e


class NotesAssistantException(Exception):
    pass
