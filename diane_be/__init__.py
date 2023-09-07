import logging
import os

import openai
import s3fs
from dotenv import load_dotenv
from flask import Flask
from langchain.chat_models import ChatOpenAI

from diane_be.assistant import NotesAssistant
from diane_be.storage import NoteStorage


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

        aws_key = os.environ['AWS_ACCESS_KEY_ID']
        aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
        r2_account_id = os.environ['R2_ACCOUNT_ID']

        assert aws_key is not None and aws_key != ""

        s3 = s3fs.S3FileSystem(
            key=aws_key,
            secret=aws_secret,
            endpoint_url=f'https://{r2_account_id}.r2.cloudflarestorage.com'
        )

        note_storage = NoteStorage(s3)

        app.config['NOTES_ASSISTANT'] = NotesAssistant(chat_llm, note_storage)
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
