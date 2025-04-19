from flask import Flask, render_template, request, jsonify
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Tool setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Tavily API setup
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(k=3, tavily_api_key=tavily_api_key)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message")
    source = data.get("source", "wiki")  # Default source

    if not query:
        return jsonify({"response": "Please ask something."})

    if source == "arxiv":
        response = arxiv.invoke(query)
    elif source == "wiki":
        response = wiki.invoke(query)
    elif source == "tavily":
        response = tavily_tool.invoke(query)
    else:
        response = "Invalid source selected."

    return jsonify({"response": response})

if __name__ =="__main__":
    app.run(debug=True)