import sample_program
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
import subprocess
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader



from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool


import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv('LANGSMITH_API_KEY')

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=API_KEY
LANGCHAIN_PROJECT="smartfile"

MODEL_CODELLAMA = "phind/phind-codellama-34b"
MODEL_LLAMA3 = "meta-llama/llama-3-70b-instruct"
MODEL_MIXTRAL22 = "mistralai/mixtral-8x22b-instruct"
MODEL_PALM2 = "google/palm-2-codechat-bison-32k"

# Initliaze models
llm_llama3 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_LLAMA3)
llm_codellama = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_CODELLAMA)
llm_mixtral22 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_MIXTRAL22)
llm_palm2 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_PALM2)

def load_documents_db(directory: str):
    documents = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        loader = TextLoader(file_path)
        if filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = BasePDFLoader(file_path)
        elif filename.endswith(".html"):    
            loader = UnstructuredHTMLLoader(file_path)

        document = loader.load()
        documents.extend(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    return db

@tool
def execute_command(command):
    """Executes a command in the shell and returns the output.

    Args:
    - command: str - The command to execute in the shell.
    """
    result = subprocess.run(command.trim().split(" "), stdout=subprocess.PIPE)
    output = {"stdout: ": result.stdout.decode('utf-8'), "stderr: ": result.stderr.decode('utf-8')}
    return output





def init_tools_agent():
    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.pretty_print()

    db = load_documents_db("llm_docs")
    retriever = db.as_retriever()

    file_tool = create_retriever_tool(
        retriever,
        "search_file_tools_docs",
        """Searches and returns excerpts from documentation for CLI usage of the following tools:
        - pandoc
        - ffmpeg
        - image magick
        """,
    )

    tools = [execute_command, file_tool]

    # Construct the tool calling agent
    agent = create_tool_calling_agent(llm_palm2, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
    {
        "input": "create an image with a small red square in the center",
    }
)
    
if __name__ == "__main__":
    init_tools_agent()
    print("Done!")