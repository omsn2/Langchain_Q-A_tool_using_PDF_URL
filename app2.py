import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfReader
import spacy
from textblob import TextBlob
from googletrans import Translator
import subprocess
import base64
from gtts import gTTS
import io
import uuid
import logging
from flask import send_from_directory

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Ensure spaCy model is downloaded
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

# Load models and embeddings once
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
Context:\n{context}\n
Question:\n{question}\n
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)

# Function to fetch text from a URL
def get_url_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error fetching the URL content: {e}")
        return f"Error fetching the URL content: {e}"

# Function to fetch text from a PDF
def get_pdf_text(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        logging.error(f"Error reading the PDF content: {e}")
        return f"Error reading the PDF content: {e}"

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to handle user input
def user_input(user_questions):
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = []
        responses = []
        for question in user_questions:
            question = question.lower()
            retrieved_docs = new_db.similarity_search(question)
            docs.append(retrieved_docs)
            response = qa_chain({"input_documents": retrieved_docs, "question": question})
            responses.append(response["output_text"])
        return docs, responses
    except Exception as e:
        logging.error(f"Error handling user input: {e}")
        return [], [f"Error: {e}"]

# Function to perform named entity recognition (NER)
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
        sentiment = blob.sentiment
        return sentiment
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return None

# Function to translate text to a specified language
def translate_text(text, target_lang='hi'):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return None

# Function to summarize text
def summarize_text(text, num_sentences=3):
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        summary = " ".join([str(sent) for sent in sentences[:num_sentences]])
        return summary
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return "Error summarizing text."

# Function to convert text to speech
def text_to_speech(text, filename):
    try:
        if not text.strip():
            raise ValueError("No text to speak")
        tts = gTTS(text)
        tts.save(filename)
        return filename
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        return None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"])
server = app.server

# Serve static files
@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Langchain Q&A using Large Language Models 💁", className="text-center mt-4", style={'color': '#272343'}),
                dcc.Tabs([
                    dcc.Tab(label='Home', children=[
                        html.Div([
                            html.H3("Welcome to the Chat Article App!", className="text-center mt-4", style={'color': '#000080'}),
                            html.P("Use the navigation tabs to switch between different functionalities.", className="text-center mt-2", style={'color': '#4B0082'}),
                            html.Ul([
                                html.Li("Process URL: Analyze and ask questions about articles from a URL.", className="mt-2", style={'color': '#4B0082'}),
                                html.Li("Process PDF: Analyze and ask questions about articles from a PDF file.", className="mt-2", style={'color': '#4B0082'})
                            ])
                        ], className="p-4 rounded shadow-sm", style={'backgroundColor': '#e3f6f5'})
                    ]),
                    dcc.Tab(label='Process URL', children=[
                        html.Div([
                            dbc.Input(id="url-input", placeholder="Enter URL of the article", type="text", className="mt-3 mb-3"),
                            dbc.Button("Submit & Process URL", id="submit-url", color="primary", className="mb-3"),
                            dcc.Loading(id="url-loading", children=[
                                html.Div(id="url-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e9ecef'})
                            ]),
                            html.Div([
                                dbc.Textarea(id="url-questions", placeholder="Ask Questions from the Article", rows=5, className="mt-3 mb-3"),
                                dbc.Select(id="url-lang-select", options=[
                                    {"label": "English", "value": "en"},
                                    {"label": "Hindi", "value": "hi"},
                                    {"label": "Spanish", "value": "es"}
                                ], value="en", className="mb-3"),
                                dbc.Input(id="url-sentences", type="number", placeholder="Number of sentences for summary", value=3, className="mb-3"),
                                dbc.Button("Process Questions", id="process-url-questions", color="success", className="mb-3")
                            ]),
                            dcc.Loading(id="questions-loading", children=[
                                html.Div(id="questions-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e9ecef'})
                            ])
                        ])
                    ]),
                    dcc.Tab(label='Process PDF', children=[
                        html.Div([
                            dcc.Upload(id="pdf-upload", children=html.Div(["Drag and Drop or ", html.A("Select a PDF File")], className="text-center mt-3 mb-3"), style={"height": "60px", "lineHeight": "60px", "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center"}, multiple=False),
                            html.Div(id="pdf-file-name", className="text-center mt-3 mb-3"),
                            dbc.Button("Process PDF", id="submit-pdf", color="primary", className="mb-3"),
                            dcc.Loading(id="pdf-loading", children=[
                                html.Div(id="pdf-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e9ecef'})
                            ]),
                            html.Div([
                                dbc.Textarea(id="pdf-questions", placeholder="Ask up to 15 Questions from the PDF (separate by semicolon)", rows=5, className="mt-3 mb-3"),
                                dbc.Select(id="pdf-lang-select", options=[
                                    {"label": "English", "value": "en"},
                                    {"label": "Hindi", "value": "hi"},
                                    {"label": "Spanish", "value": "es"}
                                ], value="en", className="mb-3"),
                                dbc.Input(id="pdf-sentences", type="number", placeholder="Number of sentences for summary", value=3, className="mb-3"),
                                dbc.Button("Process Questions", id="process-pdf-questions", color="success", className="mb-3")
                            ]),
                            dcc.Loading(id="pdf-questions-loading", children=[
                                html.Div(id="pdf-questions-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e9ecef'})
                            ])
                        ])
                    ])
                ])
            ])
        ])
    ])
])

# Callbacks for processing URL
@app.callback(
    Output("url-output", "children"),
    Input("submit-url", "n_clicks"),
    State("url-input", "value")
)
def process_url(n_clicks, url):
    if n_clicks is None or not url:
        return ""
    text = get_url_text(url)
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)
    return html.Div([
        html.H5("URL has been processed successfully. You can now ask questions.", className="mt-3")
    ])

@app.callback(
    Output("questions-output", "children"),
    Input("process-url-questions", "n_clicks"),
    State("url-questions", "value"),
    State("url-lang-select", "value"),
    State("url-sentences", "value")
)
def process_url_questions(n_clicks, questions, lang, num_sentences):
    if n_clicks is None or not questions:
        return ""
    questions_list = questions.split(";")
    if len(questions_list) > 15:
        return "Please limit to 15 questions."

    docs, responses = user_input(questions_list)
    translated_responses = [translate_text(response, lang) for response in responses]
    summaries = [summarize_text(response, num_sentences) for response in responses]

    results = []
    for i, question in enumerate(questions_list):
        results.append(html.Div([
            html.H5(f"Question {i+1}: {question.strip()}"),
            html.P(f"Answer: {translated_responses[i]}"),
            html.P(f"Summary: {summaries[i]}"),
            html.P(f"Sentiment: {analyze_sentiment(translated_responses[i])}"),
            html.P(f"NER: {perform_ner(translated_responses[i])}")
        ], className="mt-3 mb-3"))
    return results

# Callbacks for processing PDF
@app.callback(
    Output("pdf-file-name", "children"),
    Input("pdf-upload", "contents"),
    State("pdf-upload", "filename"),
    State("pdf-upload", "last_modified")
)
def update_pdf_output(contents, filename, last_modified):
    if contents is None:
        return ""
    return f"Uploaded PDF: {filename}"

@app.callback(
    Output("pdf-output", "children"),
    Input("submit-pdf", "n_clicks"),
    State("pdf-upload", "contents"),
    State("pdf-upload", "filename")
)
def process_pdf(n_clicks, contents, filename):
    if n_clicks is None or contents is None:
        return ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    text = get_pdf_text(io.BytesIO(decoded))
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)
    return html.Div([
        html.H5("PDF has been processed successfully. You can now ask questions.", className="mt-3")
    ])

@app.callback(
    Output("pdf-questions-output", "children"),
    Input("process-pdf-questions", "n_clicks"),
    State("pdf-questions", "value"),
    State("pdf-lang-select", "value"),
    State("pdf-sentences", "value")
)
def process_pdf_questions(n_clicks, questions, lang, num_sentences):
    if n_clicks is None or not questions:
        return ""
    questions_list = questions.split(";")
    if len(questions_list) > 15:
        return "Please limit to 15 questions."

    docs, responses = user_input(questions_list)
    translated_responses = [translate_text(response, lang) for response in responses]
    summaries = [summarize_text(response, num_sentences) for response in responses]

    results = []
    for i, question in enumerate(questions_list):
        results.append(html.Div([
            html.H5(f"Question {i+1}: {question.strip()}"),
            html.P(f"Answer: {translated_responses[i]}"),
            html.P(f"Summary: {summaries[i]}"),
            html.P(f"Sentiment: {analyze_sentiment(translated_responses[i])}"),
            html.P(f"NER: {perform_ner(translated_responses[i])}")
        ], className="mt-3 mb-3"))
    return results

if __name__ == '__main__':
    app.run_server(debug=True)
