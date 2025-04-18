from flask import Flask,render_template,request
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()
import os


app = Flask(__name__)

# Creating the Instance of the ArxivQueryRun and WikipediaQueryRun classes
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Setup Tavily
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(k=3,tavily_api_key=tavily_api_key) 

@app.route("/")
def home():
    result = ""
    if request.method == "POST":
        query = request.form["query"]
        source = request.form["source"]

        if source == "arxiv":
            result = arxiv.invoke(query)
        elif (source == "wiki"):
            result = arxiv.invoke(query)
        elif (source == "tavily"):
            result = tavily_tool.invoke(query)

    return render_template("index.html", result=result) 