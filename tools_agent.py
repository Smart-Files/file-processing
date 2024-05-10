import sample_program
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_core.tools import tool, Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent, AgentType, initialize_agent, load_tools
import subprocess
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.globals import set_debug, set_verbose
from langchain_core.messages import AIMessage
import shlex
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)


from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langsmith import Client


import os
from dotenv import load_dotenv


# chroma_client = Chroma.PersistentClient()

set_verbose(True)
set_debug(True)

load_dotenv()
PERSIST_DIR = 'db'

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY= os.getenv('LANGSMITH_API_KEY')
LANGCHAIN_PROJECT="smartfile"

MODEL_CODELLAMA = "phind/phind-codellama-34b"
MODEL_LLAMA3 = "meta-llama/llama-3-70b-instruct:nitro"
MODEL_LLAMA3_8B = "meta-llama/llama-3-8b-instruct:nitro"
MODEL_MIXTRAL22 = "mistralai/mixtral-8x22b-instruct"
MODEL_PALM2 = "google/palm-2-codechat-bison-32k"
MODEL_QWEN = "qwen/qwen-110b-chat"

# Initliaze models
llm_llama3 = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_LLAMA3)
llm_llama3_8b = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_LLAMA3_8B)
llm_codellama = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_CODELLAMA)
llm_mixtral22 = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_MIXTRAL22)

client = Client()


def load_documents_db(directory: str):
    documents = []


    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    try:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        if vectordb: 
            print("Found Database: Importing!")
            return vectordb
    except Exception as e:
        print("Could not load DB from disk: ", e)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        loader = TextLoader(file_path)
        if filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = BasePDFLoader(file_path)
        elif filename.endswith(".html"):    
            loader = BSHTMLLoader(file_path)

        document = loader.load()
        documents.extend(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=PERSIST_DIR)

    # vectordb.persist()



    return vectordb

@tool
def execute_command(command: str) -> dict["stdout": str, "stderr": str]:
    """Executes a command in the shell and returns the output.

    Args:
    - command: str - The command to execute in the shell.
    """


    print("start-finish:", command[0], command[len(command)-1])
    command = command.strip()
    if command[0] in ["'", "`", '"', '`'] and command[len(command)-1] == command[0]:
        command = command[1:len(command)-1]

    args = shlex.split(command)
    print("ARGS: ", args)
    result = subprocess.run(args, stdout=subprocess.PIPE)

    print("RESULT: ", result)

    stdout = None
    if (result.stdout):
        stdout = result.stdout.decode('utf-8')

    stderr = None
    if (result.stderr):
        stderr = result.stderr.decode('utf-8')
    
    output = {"stdout": stdout, "stderr": stderr}
    print(output)
    return output





def init_tools_agent(llm, input, output_file):
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

        Search any query with the tool name to get the documentation for that tool. For example, "pandoc convert markdown to pdf" will return the documentation for the pandoc tool.
        """,
    )


    execute_tool = Tool(name="Execute Shell Command",func=execute_command, description="Executes a shell command and returns the output.")




    tools = [execute_tool, file_tool]

    # Construct the tool calling agent
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, )

    agent.handle_parsing_errors = True
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


    output = agent.invoke({
        "input": f'{input} The output file should be: {output_file}',
    })

    print(output)
    
if __name__ == "__main__":
    input = "downscale stick_figure.png to 10px by 10px and then add it three times to a pdf file and then convert that to docx"

    init_tools_agent(llm_llama3, input, "stick_figure.docx")
    # init_tools_agent(llm_llama3_8b, input, "llama8b.pdf")
    # init_tools_agent(llm_codellama, input, "codellama.png")
    # init_tools_agent(llm_mixtral22, input, "mixtral.png")
    # init_tools_agent(llm_palm2, input, "palm2.png")
    # init_tools_agent(llm_qwen, input, "qwen.png")
    print("Done!")