from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
import subprocess
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import BasePDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langsmith import Client

from fileprocessing import execute_command
from fileprocessing import tool_doc_retrieval


import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
PERSIST_DIR = 'db'

OPENROUTER_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY= os.getenv('LANGSMITH_API_KEY')
LANGCHAIN_PROJECT="smartfile"

MODEL_CODELLAMA = "phind/phind-codellama-34b"
MODEL_LLAMA3 = "llama3-70b-8192"
MODEL_LLAMA3_8B = "meta-llama/llama-3-8b-instruct:nitro"
MODEL_MIXTRAL22 = "mistralai/mixtral-8x22b-instruct"
MODEL_PALM2 = "google/palm-2-codechat-bison-32k"
MODEL_QWEN = "qwen/qwen-110b-chat"

API_BASE = "https://api.groq.com/openai/v1"
# API_BASE = "https://openrouter.ai/api/v1"



# Initliaze models
llm_llama3 = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base=API_BASE, model_name=MODEL_LLAMA3)
# llm_llama3_8b = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base=API_BASE, model_name=MODEL_LLAMA3_8B)
# llm_codellama = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base=API_BASE, model_name=MODEL_CODELLAMA)
# llm_mixtral22 = ChatOpenAI(openai_api_key=OPENROUTER_API_KEY, openai_api_base=API_BASE, model_name=MODEL_MIXTRAL22)

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=PERSIST_DIR)

    # vectordb.persist()



    return vectordb

    

async def init_tools_agent( uuid: str) -> AgentExecutor:
    """
    Generates tools and prompts and return them along with the preferred model
    """
    prompt = hub.pull("hwchase17/react")

    file_tool = tool_doc_retrieval.create_file_retrieval_tool()


    execute_tool = Tool(name="Execute Shell Command",func=execute_command.execute_command_factory(uuid), description="Executes a shell command and returns the output. Do not install any software or packages, all packages are available using tools in documentation")


    tools = [execute_tool, file_tool]

    # Construct the tool calling agent
    # agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, callback_manager=callback_manager)


    # async for s in agent_executor.astream_log({"input": f'{input}\nThe output file should be: {output_file}'}):
    #     print(s, end="e")

    # print(output)

    return {"llm": llm_llama3, "tools": tools, "prompt": prompt}
    
if __name__ == "__main__":
    input = ""
    asyncio.run(init_tools_agent(llm_llama3, input, "mathoutput.pdf"))
    # init_tools_agent(llm_llama3_8b, input, "llama8b.pdf")
    # init_tools_agent(llm_codellama, input, "codellama.png")
    # init_tools_agent(llm_mixtral22, input, "mixtral.png")
    # init_tools_agent(llm_palm2, input, "palm2.png")
    # init_tools_agent(llm_qwen, input, "qwen.png")
    print("Done!")