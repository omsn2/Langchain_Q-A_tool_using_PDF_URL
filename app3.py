import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import os
import logging
from functools import lru_cache
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import spacy
from textblob import TextBlob
from gtts import gTTS
import base64
import uuid
import subprocess  # Added missing import for subprocess
from flask import send_from_directory
import google.generativeai as genai  # Added missing import for genai

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

genai.configure(api_key=google_api_key)

# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Initialize models and embeddings once
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, max_tokens=100)
prompt_template = """
Answer the question as detailed as possible from the provided context. If the answer is not in the context, say "Answer not available."
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = LLMChain(llm=chat_model, prompt=prompt)

# Load vector store once
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Daily rate limiter setup
query_count = 0
reset_time = datetime.now() + timedelta(days=1)

# Caching for optimized response
@lru_cache(maxsize=20)
def cached_user_input(user_question):
    docs = vector_store.similarity_search(user_question)
    response = qa_chain({"input_documents": docs, "question": user_question})
    return docs, response["output_text"]

# Function to handle rate-limited input
def rate_limited_user_input(user_question):
    global query_count, reset_time
    if datetime.now() >= reset_time:
        query_count = 0
        reset_time = datetime.now() + timedelta(days=1)
    if query_count >= 10:
        return [], "Daily query limit reached. Try again tomorrow."
    query_count += 1
    return cached_user_input(user_question)

# Function to fetch text from URL
def get_url_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error fetching URL content: {e}")
        return f"Error: {e}"

# Function to fetch text from PDF
def get_pdf_text(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        return text
    except Exception as e:
        logging.error(f"Error reading PDF content: {e}")
        return f"Error: {e}"

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to perform NER
def perform_ner(text):
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        logging.error(f"Error performing NER: {e}")
        return []

# Function to perform sentiment analysis
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return None

# Function to summarize text
def summarize_text(text, num_sentences=3):
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        return " ".join([str(sent) for sent in sentences[:num_sentences]])
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return "Error summarizing text."

# Function to convert text to speech
def text_to_speech(text):
    try:
        tts = gTTS(text)
        filename = f"static/{uuid.uuid4()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        return None

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Serve static files
@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Layout
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Langchain Q&A with Daily Rate Limit 💁", className="text-center mt-4"),
                dcc.Tabs([
                    dcc.Tab(label='Process URL', children=[
                        html.Div([
                            dbc.Input(id="url-input", placeholder="Enter URL of the article", type="text"),
                            dbc.Button("Submit & Process URL", id="submit-url", color="primary"),
                            dcc.Loading(id="url-loading", children=[html.Div(id="url-output")]),
                            dbc.Input(id="url-question", placeholder="Ask a Question from the Article"),
                            dbc.Button("Submit Question", id="submit-url-question", color="primary"),
                            dcc.Loading(id="url-question-loading", children=[html.Div(id="url-question-output")]),
                            html.Audio(id="url-audio", controls=True, className="mt-3")
                        ])
                    ]),
                    dcc.Tab(label='Process PDF', children=[
                        html.Div([
                            dcc.Upload(
                                id='upload-pdf',
                                children=html.Div(['Drag and Drop or ', html.A('Select PDF File')]),
                                style={'borderStyle': 'dashed'}
                            ),
                            dcc.Loading(id="pdf-loading", children=[html.Div(id="pdf-output")]),
                            dbc.Input(id="pdf-question", placeholder="Ask a Question from the PDF"),
                            dbc.Button("Submit Question", id="submit-pdf-question", color="primary"),
                            dcc.Loading(id="pdf-question-loading", children=[html.Div(id="pdf-question-output")]),
                            html.Audio(id="pdf-audio", controls=True, className="mt-3")
                        ])
                    ])
                ])
            ])
        ])
    ])
])

# Callbacks
@app.callback(
    Output("url-output", "children"),
    Input("submit-url", "n_clicks"),
    State("url-input", "value")
)
def process_url(n_clicks, url):
    if n_clicks and url:
        logging.info(f"Processing URL: {url}")
        raw_text = get_url_text(url)
        if raw_text.startswith("Error"):
            return html.Div(raw_text, className="text-danger")
        text_chunks = get_text_chunks(raw_text)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Updated vector store in memory
        return html.Div("Processing complete!", className="text-success")
    return ""

@app.callback(
    [Output("url-question-output", "children"), Output("url-audio", "src")],
    Input("submit-url-question", "n_clicks"),
    State("url-question", "value")
)
def handle_url_question(n_clicks, user_question):
    if n_clicks and user_question:
        logging.info(f"User question: {user_question}")
        docs, response = rate_limited_user_input(user_question)
        response_text = response if response else "No answer found."
        points = response_text.split('. ')
        formatted_response = html.Ul([html.Li(point) for point in points if point])

        audio_file = text_to_speech(response_text)
        audio_src = f"/static/{os.path.basename(audio_file)}" if audio_file else None
        return [formatted_response, audio_src]
    return [None, None]

@app.callback(
    Output("pdf-output", "children"),
    Input('upload-pdf', 'contents'),
    State('upload-pdf', 'filename')
)
def process_pdf(contents, filename):
    if contents:
        logging.info(f"Processing PDF: {filename}")
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path, 'wb') as f:
            f.write(decoded)
        
        raw_text = get_pdf_text(file_path)
        if raw_text.startswith("Error"):
            return html.Div(raw_text, className="text-danger")
        text_chunks = get_text_chunks(raw_text)
        FAISS.from_texts(text_chunks, embedding=embeddings)
        return html.Div("Processing complete!", className="text-success")
    return ""

if __name__ == "__main__":
    app.run_server(debug=True)
