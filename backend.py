from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # CHANGED: replaces ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings — Gemini fully removed
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import math
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os 
import json
from typing import Any
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig


load_dotenv()


# Where runtime state is written: the SQLite checkpointer and the FAISS
# indexes. Unset locally, so it defaults to "." and behaves exactly as
# before. In Docker it is set to /app/data, which is a mounted volume, so
# conversations and indexes survive the container being replaced on deploy.
DATA_DIR = os.getenv("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)


# LLM
# CHANGED: gemini-2.5-flash -> gpt-5-nano. Gemini access kept breaking
# (404s on new API keys, then garbled streaming output from a content-
# block-list response format). gpt-5-nano's streamed chunks are plain
# strings, so `yield message_chunk.content` in ai_only_stream() works
# as written. Needs OPENAI_API_KEY in your .env. If this errors on
# `temperature`, delete that kwarg — some reasoning-family OpenAI
# models only accept the default temperature.
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.7
)


# Embeddings model
# CHANGED: was GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
# -> now OpenAIEmbeddings, so nothing in this file touches Gemini/Google
# anymore. GOOGLE_API_KEY is no longer needed at all; OPENAI_API_KEY now
# covers both the chat model and embeddings.
#
# IMPORTANT: this changes the vector space entirely. If you already have
# a faiss_db/ folder from before this change, it was built with Gemini's
# embeddings and is now INCOMPATIBLE — delete that folder and re-upload
# your PDF once so it gets re-indexed with OpenAI's embeddings. Mixing
# embedding models between indexing and querying silently produces
# meaningless similarity results rather than an obvious error.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



def _index_path(thread_id=None):
    """
    Vector index directory for ONE conversation.

    The index used to be a single global "faiss_db" folder, which meant a
    second upload silently overwrote the first and every user of a deployed
    instance shared one index. Scoping by thread_id isolates conversations.
    """
    return os.path.join(
        DATA_DIR, "faiss_db", str(thread_id) if thread_id else "default"
    )


def _manifest_path(thread_id=None):
    """JSON list of document names indexed for this conversation."""
    return os.path.join(_index_path(thread_id), "documents.json")


def list_indexed_documents(thread_id=None):
    """Names of documents currently indexed for this conversation."""
    path = _manifest_path(thread_id)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return []


def ingest_rag_document(file_path, thread_id=None, display_name=None):
    """Chunk, embed and index a PDF into this conversation's vector store."""
    db_path = _index_path(thread_id)

    # The frontend indexes a NamedTemporaryFile, so metadata["source"] would
    # otherwise be an unreadable temp path. Keep the name the user uploaded.
    name = display_name or os.path.basename(file_path)

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = name

    if os.path.isdir(db_path):
        # Add to the existing index rather than replacing it, so uploading a
        # second PDF does not destroy the first.
        vector_store = FAISS.load_local(
            folder_path=db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(db_path)

    # Record the name so chat_node can tell the model what is available.
    names = list_indexed_documents(thread_id)
    if name not in names:
        names.append(name)
    with open(_manifest_path(thread_id), "w", encoding="utf-8") as handle:
        json.dump(names, handle)
    


def get_retriever(thread_id=None):
    """Return a retriever for this conversation, or None if nothing indexed."""
    db_path = _index_path(thread_id)

    # No PDF uploaded in this conversation yet - callers must handle None.
    if not os.path.isdir(db_path):
        return None

    vector_store = FAISS.load_local(
            folder_path=db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever




# rag tool

@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    # `config` is injected by LangChain and hidden from the model's schema.
    thread_id = (config or {}).get("configurable", {}).get("thread_id")

    retriever = get_retriever(thread_id)

    if retriever is None:
        return (
            "No PDF has been uploaded in this conversation yet. "
            "Ask the user to upload a PDF using the attachment button in the "
            "chat input, then try the question again."
        )

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        # PyPDFLoader numbers pages from zero; humans do not.
        raw_page = document.metadata.get("page")
        page = raw_page + 1 if isinstance(raw_page, int) else "Unknown page"

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)




# Tools

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    # CHANGED: key was hardcoded here. Never commit a live key to a public
    # repo - read it from the environment (.env) instead.
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {
            "error": "Stock API key is missing. "
                     "Set the ALPHA_VANTAGE_API_KEY environment variable."
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    r = requests.get(url, timeout=15)
    return r.json()



@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }




@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"
    


# Make tool list
tools = [search_tool,calculator, get_stock_price,get_current_weather, rag_tool, purchase_stock]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)




# State
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]



# Nodes 1
def chat_node(state: ChatState, config: RunnableConfig = None):
    """LLM node that can answer directly or call an appropriate tool."""

    # Which documents exist is per-conversation and changes mid-conversation,
    # so it must be injected each turn rather than hardcoded in the prompt.
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    indexed = list_indexed_documents(thread_id)

    if indexed:
        document_status = (
            "DOCUMENTS CURRENTLY INDEXED IN THIS CONVERSATION: "
            + ", ".join(indexed) + ".\n"
            "Search them with `rag_tool`. The user can upload more at any time, "
            "so this list changes between turns. You MUST call `rag_tool` in the "
            "current turn for any question these documents might cover. Never "
            "answer from what you remember of earlier tool results, and never "
            "state that a topic is absent from the documents unless you have "
            "searched for it in THIS turn.\n\n"
        )
    else:
        document_status = (
            "NO DOCUMENTS ARE INDEXED IN THIS CONVERSATION YET. If the user asks "
            "about an uploaded document, tell them to upload a PDF first.\n\n"
        )


    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            + document_status +

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `purchase_stock` when the user wants to purchase a stock. "
            "Call the tool immediately. Do NOT ask the user to confirm first, "
            "and never tell them to reply 'yes' or 'no' in chat. The system "
            "pauses the purchase and collects approval through a separate "
            "approval control. Asking for confirmation yourself creates a "
            "second, fake approval step that the user cannot distinguish "
            "from the real one.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}




# Nodes 2 - tool node
tool_node = ToolNode(tools)



# Checkpointer
conn = sqlite3.connect(
    database=os.path.join(DATA_DIR, "chatbot.db"),
    check_same_thread=False,
)
checkpoint = SqliteSaver(conn)



# graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)



# Helper functions for Streamlit frontend
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)


