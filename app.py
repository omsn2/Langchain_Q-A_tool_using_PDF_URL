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
import pandas as pd
import re
from PyPDF2 import PdfReader
import graphviz
import spacy
from textblob import TextBlob
from googletrans import Translator
import subprocess
import base64

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
        return f"Error reading the PDF content: {e}"

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return docs, response["output_text"]

# Function to extract numeric data from text
def extract_numeric_data(text):
    pattern = r'([A-Za-z\s&]+)\s(\d{1,3}(?:,\d{3})(?:\.\d+)?)(?:\s(\d{1,3}(?:,\d{3})(?:\.\d+)?))?'
    matches = re.findall(pattern, text)
    data = {}
    for match in matches:
        category = match[0].strip()
        value_2024 = float(match[1].replace(',', ''))
        value_2023 = float(match[2].replace(',', '')) if match[2] else None
        
        if category not in data:
            data[category] = {"2024": [], "2023": []}
        data[category]["2024"].append(value_2024)
        if value_2023 is not None:
            data[category]["2023"].append(value_2023)
    return data

# Function to generate graphs based on user input
def generate_graphs(data, user_graph_request):
    if not data:
        return "No numeric data available to generate graphs."

    df = pd.DataFrame(data)

    if not df.empty:
        fig = None
        if user_graph_request and "growth" in user_graph_request.lower():
            df_growth = df["2024"] - df["2023"]
            fig = df_growth.plot.line().get_figure()
        elif user_graph_request and "sales" in user_graph_request.lower():
            df_sales = pd.DataFrame({
                '2024': df["2024"],
                '2023': df["2023"]
            })
            fig = df_sales.plot.line().get_figure()
        else:
            fig = df.plot.line().get_figure()

        return fig
    else:
        return "No specific graph requested or data available for the request."

# Function to generate flowcharts
def generate_flowchart(context):
    dot = graphviz.Digraph()
    sentences = context.split('.')
    for i, sentence in enumerate(sentences):
        dot.node(str(i), sentence.strip())
        if i > 0:
            dot.edge(str(i-1), str(i))
    return dot

# Function to perform named entity recognition (NER)
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# Function to translate text to a specified language
def translate_text(text, target_lang='hi'):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text

# Function to summarize text
def summarize_text(text, num_sentences=3):
    doc = nlp(text)
    sentences = list(doc.sents)
    summary = " ".join([str(sent) for sent in sentences[:num_sentences]])
    return summary

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Chat with Article using Gemini 💁", className="text-center mt-4"),
                dcc.Tabs([
                    dcc.Tab(label='Home', children=[
                        html.Div([
                            html.H3("Welcome to the Chat Article app!"),
                            html.P("Use the navigation tabs to switch between different functionalities."),
                            html.Ul([
                                html.Li("Process URL: Analyze and ask questions about articles from a URL."),
                                html.Li("Process PDF: Analyze and ask questions about articles from a PDF file.")
                            ])
                        ])
                    ]),
                    dcc.Tab(label='Process URL', children=[
                        html.Div([
                            dbc.Input(id="url-input", placeholder="Enter URL of the article", type="text", className="mt-3"),
                            dbc.Button("Submit & Process URL", id="submit-url", color="primary", className="mt-2"),
                            dcc.Loading(id="url-loading", children=[
                                html.Div(id="url-output")
                            ]),
                            html.Div([
                                dbc.Input(id="url-question", placeholder="Ask a Question from the Article", type="text", className="mt-3"),
                                dbc.Input(id="url-graph-request", placeholder="Request a specific graph (e.g., 'growth', 'sales')", type="text", className="mt-3"),
                                dbc.Select(id="url-translate", options=[
                                    {'label': 'Hindi', 'value': 'hi'},
                                    {'label': 'Bengali', 'value': 'bn'},
                                    {'label': 'Telugu', 'value': 'te'},
                                    {'label': 'Marathi', 'value': 'mr'},
                                    {'label': 'Tamil', 'value': 'ta'},
                                    {'label': 'Gujarati', 'value': 'gu'},
                                    {'label': 'Kannada', 'value': 'kn'},
                                    {'label': 'Malayalam', 'value': 'ml'},
                                    {'label': 'Odia', 'value': 'or'},
                                    {'label': 'Punjabi', 'value': 'pa'}
                                ], placeholder="Translate text to:", className="mt-3"),
                                dbc.Button("Submit Question", id="submit-url-question", color="primary", className="mt-2"),
                                dcc.Loading(id="url-question-loading", children=[
                                    html.Div(id="url-question-output")
                                ])
                            ])
                        ])
                    ]),
                    dcc.Tab(label='Process PDF', children=[
                        html.Div([
                            dcc.Upload(
                                id='upload-pdf',
                                children=html.Div(['Drag and Drop or ',
                                    html.A('Select PDF File')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            dcc.Loading(id="pdf-loading", children=[
                                html.Div(id="pdf-output")
                            ]),
                            html.Div([
                                dbc.Input(id="pdf-question", placeholder="Ask a Question from the Article", type="text", className="mt-3"),
                                dbc.Input(id="pdf-graph-request", placeholder="Request a specific graph (e.g., 'growth', 'sales')", type="text", className="mt-3"),
                                dbc.Select(id="pdf-translate", options=[
                                    {'label': 'Hindi', 'value': 'hi'},
                                    {'label': 'Bengali', 'value': 'bn'},
                                    {'label': 'Telugu', 'value': 'te'},
                                    {'label': 'Marathi', 'value': 'mr'},
                                    {'label': 'Tamil', 'value': 'ta'},
                                    {'label': 'Gujarati', 'value': 'gu'},
                                    {'label': 'Kannada', 'value': 'kn'},
                                    {'label': 'Malayalam', 'value': 'ml'},
                                    {'label': 'Odia', 'value': 'or'},
                                    {'label': 'Punjabi', 'value': 'pa'}
                                ], placeholder="Translate text to:", className="mt-3"),
                                dbc.Button("Submit Question", id="submit-pdf-question", color="primary", className="mt-2"),
                                dcc.Loading(id="pdf-question-loading", children=[
                                    html.Div(id="pdf-question-output")
                                ])
                            ])
                        ])
                    ])
                ])
            ])
        ])
    ])
])

@app.callback(
    Output("url-output", "children"),
    Input("submit-url", "n_clicks"),
    State("url-input", "value")
)
def process_url(n_clicks, url):
    if n_clicks and url:
        raw_text = get_url_text(url)
        if raw_text.startswith("Error"):
            return html.Div(raw_text, className="text-danger")
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return html.Div("Processing complete!", className="text-success")
    return ""

@app.callback(
    Output("url-question-output", "children"),
    Input("submit-url-question", "n_clicks"),
    State("url-question", "value"),
    State("url-graph-request", "value"),
    State("url-translate", "value")
)
def handle_url_question(n_clicks, user_question, user_graph_request, target_lang):
    if n_clicks and user_question:
        docs, response = user_input(user_question)
        outputs = [html.Div(f"Reply: {response}", className="mt-2")]
        
        if user_graph_request:
            data = extract_numeric_data(" ".join([doc.page_content for doc in docs]))
            fig = generate_graphs(data, user_graph_request)
            if isinstance(fig, str):
                outputs.append(html.Div(fig, className="text-warning"))
            else:
                outputs.append(dcc.Graph(figure=fig))
        
        if "flowchart" in user_question.lower():
            flowchart = generate_flowchart(response)
            outputs.append(dcc.Graph(id="flowchart", figure=flowchart))
        
        if "translate" in user_question.lower():
            translation = translate_text(response, target_lang)
            outputs.append(html.Div(f"Translation: {translation}", className="mt-2"))
        
        if "ner" in user_question.lower():
            entities = perform_ner(response)
            outputs.append(html.Div(f"Named Entities: {entities}", className="mt-2"))
        
        if "summarize" in user_question.lower():
            summary = summarize_text(response)
            outputs.append(html.Div(f"Summary: {summary}", className="mt-2"))
        
        if "sentiment" in user_question.lower():
            sentiment = analyze_sentiment(response)
            outputs.append(html.Div(f"Sentiment Analysis: {sentiment}", className="mt-2"))
        
        return outputs
    return ""

@app.callback(
    Output("pdf-output", "children"),
    Input('upload-pdf', 'contents'),
    State('upload-pdf', 'filename')
)
def process_pdf(contents, filename):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Use the current working directory or specify a path
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path, 'wb') as f:
            f.write(decoded)
        
        raw_text = get_pdf_text(file_path)
        if raw_text.startswith("Error"):
            return html.Div(raw_text, className="text-danger")
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return html.Div("Processing complete!", className="text-success")
    return ""

@app.callback(
    Output("pdf-question-output", "children"),
    Input("submit-pdf-question", "n_clicks"),
    State("pdf-question", "value"),
    State("pdf-graph-request", "value"),
    State("pdf-translate", "value")
)
def handle_pdf_question(n_clicks, user_question, user_graph_request, target_lang):
    if n_clicks and user_question:
        docs, response = user_input(user_question)
        outputs = [html.Div(f"Reply: {response}", className="mt-2")]
        
        if user_graph_request:
            data = extract_numeric_data(" ".join([doc.page_content for doc in docs]))
            fig = generate_graphs(data, user_graph_request)
            if isinstance(fig, str):
                outputs.append(html.Div(fig, className="text-warning"))
            else:
                outputs.append(dcc.Graph(figure=fig))
        
        if "flowchart" in user_question.lower():
            flowchart = generate_flowchart(response)
            outputs.append(dcc.Graph(id="flowchart", figure=flowchart))
        
        if "translate" in user_question.lower():
            translation = translate_text(response, target_lang)
            outputs.append(html.Div(f"Translation: {translation}", className="mt-2"))
        
        if "ner" in user_question.lower():
            entities = perform_ner(response)
            outputs.append(html.Div(f"Named Entities: {entities}", className="mt-2"))
        
        if "summarize" in user_question.lower():
            summary = summarize_text(response)
            outputs.append(html.Div(f"Summary: {summary}", className="mt-2"))
        
        if "sentiment" in user_question.lower():
            sentiment = analyze_sentiment(response)
            outputs.append(html.Div(f"Sentiment Analysis: {sentiment}", className="mt-2"))
        
        return outputs
    return ""

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)

                                   
