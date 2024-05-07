## Import necessary libraries
import os
import requests, json
from dotenv import load_dotenv
import re
import zipfile
import io
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
load_dotenv()

API_KEY = os.getenv('OPENROUTER_API_KEY')

MODEL_CODELLAMA = "phind/phind-codellama-34b"
MODEL_LLAMA3 = "meta-llama/llama-3-70b-instruct"
MODEL_WIZARD8x22b = "microsoft/wizardlm-2-8x22b"

# Initliaze models
llm_llama3 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_LLAMA3)
llm_codellama = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_CODELLAMA)
llm_wizard22 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_WIZARD8x22b)

def get_zip_structure(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_structure = io.StringIO()
        zip_ref.printdir(file=zip_structure)
        structure_text = zip_structure.getvalue().strip()
    return structure_text

INSTRUCT_WRAP_CODE = "\nWrap your code in ```python\n{CODE}\n```"


input = "combine the images in this zip into a pdf file with one image as each page."

file = "images.zip"

# Generate detailed prompt from simple user prompt
template_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful prompting assistant.\
    You must write instructions for creating a python script for completing the below user prompt.\
    Your job is to expand on the instructions below to make sure there is a defined plan of execution."""),
    ("user", "User Prompt: {input}\nFile: {file}\nFile Contents: {contents}")
])

template_code = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant. Write code based on the instructions with redundancy for errors."),
    ("user", "Instructions: {input}\nFile: {file}\nFile Contents: {contents}")
])

file_contents = "\nContents of unzipped file:\n" + get_zip_structure(file)

def gen_content(llm_prompt, llm_code):
    prompt_chain = template_prompt | llm_prompt | output_parser
    
    code_chain = template_code | llm_code | output_parser
    
    prompt_output = prompt_chain.invoke({"input": input, "file": file, "contents": file_contents})
    
    prompt_output += INSTRUCT_WRAP_CODE
    
    code_output = code_chain.invoke({"input": prompt_output, "file": file, "contents": file_contents})
    
    python_code = re.search(r'```python(?s:(.*?))```', code_output).group(1)

    return python_code

code = gen_content(llm_llama3, llm_llama3)

template_requirements = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful coding assistent. Append code to the code to use pip or pip3 install imported libraries in the code"),
     ("user", "Code: {code}")])

# give the outputted code into a template and have the template append to a requirements.txt and install requirements.txt
def install_requirements(llm_prompt, llm_code):
    requirements_chain = template_prompt | llm_prompt | output_parser
    
    code_output = requirements_chain.invoke({"code": code})
    python_code = re.search(r'```python(?s:(.*?))```', code_output).group(1)

    return python_code

code = install_requirements(llm_llama3, llm_llama3)


def run_code_and_remove_errors(llm_prompt, llm_code):
    # execute code here and save the stdout/stderr. if there are any errors at compilation, save them and give them to a template
    # run the outputted code first
    # then make new template if the code throws an error
    # template_debug
    # try to get the error console throw some piping stdout/stderr and give it to the template

    # TODO: run code and see error
    error = "PLACE_HOLDER"
    # check if error
    while(error != ""):
        # while error exists/is not empty
        template_error = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful coding assistant. You have been given the following code and error outputs. Output a modified version of the code that doesn't contain any errors."),
            ("user", "Code: {code}\n Error Output: {error}")
        ])
    



