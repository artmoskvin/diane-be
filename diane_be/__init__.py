import logging
import os

import openai
from dotenv import load_dotenv
from flask import Flask
from langchain.chat_models import ChatOpenAI
from llama_index import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR

from diane_be.assistant import NotesAssistant


def create_app(test_config=None):
    logging.basicConfig(level=logging.INFO)

    load_dotenv()

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)

        chat_llm = ChatOpenAI(temperature=0)

        if os.path.exists(DEFAULT_PERSIST_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=DEFAULT_PERSIST_DIR)
            index = load_index_from_storage(storage_context)
        else:
            index = VectorStoreIndex([])

        app.config['NOTES_ASSISTANT'] = NotesAssistant(chat_llm, index)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import notes_bp
    app.register_blueprint(notes_bp.bp)

    return app
