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
    label = "Structure of the zip file:\n"
    return label + structure_text

def get_image_metadata(image_file_path):
    import PIL.Image
    with PIL.Image.open(image_file_path) as img:
        width, height = img.size
        format = img.format
        mode = img.mode
    return f"Dimensions: {width}x{height}, Format: {format}, Mode: {mode}"

def getFileContents(file):
    if os.path.exists(file):
        if file.endswith('.zip'):
            return get_zip_structure(file)
        elif file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            return get_image_metadata(file)
    return None


def createPromptTemplate(input, file):
    # Generate detailed prompt from simple user prompt
    template_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful prompting assistant. You must write instructions for creating a python script for completing the below user prompt."),
        ("user", """Please expand on the below User Prompt with detailed instructions for a Python script that can complete the task. 
         You must not write any code as you are only a prompting assistant.

    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

    You should only respond in JSON format as described below 
    Response Format: 
    {
        "thoughts": {
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- overall plan",
            "criticism": "constructive self-criticism",
            "speak": "thoughts summary to say to user",
        }
    }

    Ensure the response can be parsed by Python json.loads
        
    User Prompt: {input}
    File: {file}
    File Contents: {contents}""")
    ])


INSTRUCT_WRAP_CODE = "\nWrap your code in ```python\n{CODE}\n```"


input = "combine the images in this zip into a pdf file with one image as each page."

file = "images.zip"

template_code = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant. Write code based on the instructions with redundancy for errors."),
    ("user", "Instructions: {input}\nFile: {file}\nFile Contents: {contents}")
])


def gen_content(llm_prompt, llm_code):
    file_contents = getFileContents()

    prompt_chain = template_prompt | llm_prompt | output_parser
    
    code_chain = template_code | llm_code | output_parser
    
    prompt_output = prompt_chain.invoke({"input": input, "file": file, "contents": file_contents})
    
    prompt_output += INSTRUCT_WRAP_CODE
    
    code_output = code_chain.invoke({"input": prompt_output, "file": file, "contents": file_contents})
    
    python_code = re.search(r'```python(?s:(.*?))```', code_output).group(1)

    return python_code

code = gen_content(llm_llama3, llm_llama3)

template_requirements = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful coding assistant"),
     ("user", "Step one: Create a list of every library that is used in the below program as well as its version. Step two: Append sh commands and use pip to install all imported, or used libraries in the code. (Wrap in ```sh```)\n\n\n$ # python library imports\n$")])

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
    



