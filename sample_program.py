## Import necessary libraries
import os
import requests, json
from dotenv import load_dotenv
import re
import zipfile
import subprocess
import io
from langchain.agents import Tool, Agent, AgentExecutor, initialize_agent, load_tools, create_openapi_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.output_parsers import StrOutputParser
import sys
from io import StringIO
import contextlib
from typing import Dict, Union, Optional


output_parser = StrOutputParser()
load_dotenv()

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL_CODELLAMA = "phind/phind-codellama-34b"
MODEL_LLAMA3 = "meta-llama/llama-3-70b-instruct"
MODEL_MIXTRAL22 = "mistralai/mixtral-8x22b-instruct"
MODEL_PALM2 = "google/palm-2-codechat-bison-32k"

# Initliaze models
llm_llama3 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_LLAMA3)
llm_codellama = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_CODELLAMA)
llm_mixtral22 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_MIXTRAL22)
llm_palm2 = ChatOpenAI(openai_api_key=API_KEY, openai_api_base="https://openrouter.ai/api/v1", model_name=MODEL_PALM2)

def get_zip_structure(zip_file_path):
    """
    Get the directory structure of a zip file as a string.

    Args:
    zip_file_path (str): The path to the zip file.
    """

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_structure = io.StringIO()
        zip_ref.printdir(file=zip_structure)
        structure_text = zip_structure.getvalue().strip()
    label = "Structure of the zip file:\n"
    return label + structure_text


def get_image_metadata(image_file_path):
    """
    Get metadata of an image file.

    Args:
    image_file_path (str): The path to the image file.
    """
    import PIL.Image
    with PIL.Image.open(image_file_path) as img:
        width, height = img.size
        format = img.format
        mode = img.mode
    return f"Dimensions: {width}x{height}, Format: {format}, Mode: {mode}"


def get_file_contents(file):
    """
    Get the contents of a file.

    * If the file is a zip file, return the directory structure.
    * If the file is an image, return the metadata.
    * If the file is human-readable, return the first 300 characters of the content.

    Args:
    file (str): The path to the file.
    """
    if os.path.exists(file):
        if file.endswith('.zip'):
            return get_zip_structure(file)
        elif file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            return get_image_metadata(file)
        else:
            with open(file, 'r') as f:
                contents = f.read()
            if contents.isprintable():
                if len(contents) > 300:
                    contents = contents[:300]
                return contents
            else:
                return None
    return None


@tool
def execute_code(code: str, verbose: bool = False) -> Dict[str, Union[str, Optional[str]]]:
    """
    Executes the code and returns the result or error
    """
    with stdoutIO() as s:
        try:
            exec(code)
        except Exception as e:
            return {"error": str(e), "output": s.getvalue()}
    return {"output": s.getvalue()}


def create_prompt_template():
    """
    Returns a template for expanding the user prompt with additional instructions.
    """

    template_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful prompting assistant. Your task is to write instructions for creating a Python script for completing below user prompt."),
        ("user", """Please expand on the below User Prompt with detailed instructions for a Python script that can complete the task. 
         You must not write any code as you are only a prompting assistant.

    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

    You should only respond in JSON format as described below 
    Response Format: 
    {{
        "thoughts": {{
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- overall plan",
            "criticism": "constructive self-criticism",
            "speak": "thoughts summary to say to user",
        }}
    }}

    Ensure the response can be parsed by Python json.loads
        
    User Prompt: {input}
    File: {file}
    File Contents: {contents}""")])

    return template_prompt


def create_code_template():
    # Generate detailed prompt from simple user prompt
    template_code = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding assistant. Write code based on the instructions with redundancy for errors. If possible, try to utilize only libraries included in the requirements file contents. Only use libraries not already included in the requirements if essential."),
        ("user", """Please write a Python script that can complete the task based on the detailed instructions provided below. Ensure that the code runs on Python Version 3.11.7
    
    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

    Wrap your code in ```python\n{{CODE}}\n```
         
    Instructions: {input}
    File: {file}
    File Contents: {contents}
    Requirements File Contents: {requirements}
         """)])

    return template_code


def install_requirements(code, llm=llm_palm2):
    """
    Installs requirements.txt ONLY if requirements has been changed.
    
    """
    file = open("requirements.txt", 'r')
    contents = file.read()
    print("CONTENTS OF REQUIREMENTS:\n", contents)
    template_requirements = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful coding assistant"),
     ("user", """
      Step one: Create a list of every library that is used in the below program as well as its version. 
      Step two: Create a list of every library included in the contents of the requirements file: {requirements}
      Step three: Manually check if there are any libraries imported in the code that aren't included in the contents of the requirements. 
      Step four: Use sh commands and pip to install libraries not included in requirements but included in the code. (Only save libraries that weren't already included in requirements) 
      (example: if the code requires Pillow and fpdf but Pillow is already in requirements but fpdf is not in requirements, output ```sh pip install fpdf```)
      (Wrap in ```sh```)\n\n\n$ # python library imports\n$\n\n\n{code}.""")])


    requirements_chain = template_requirements | llm | output_parser
    code_output = requirements_chain.invoke({"code": code, "requirements": contents})
    if "```sh" in code_output:
        extracted_commands = re.search(r'```sh(?s:(.*?))```', code_output)
        if extracted_commands:
            sh_code = extracted_commands.group(1)
        else:
            sh_code = None
        sh_code = sh_code.split("\n")

        # Remove any invalid commands such as comments or empty strings
        sh_code = [command for command in sh_code if (('#' not in command) and ( command != ''))]
        result = subprocess.run(sh_code, stdout=subprocess.PIPE, shell=True, text=True, stderr=subprocess.PIPE)
        installs = [install.replace("pip install", "") for install in sh_code]
        file = open("requirements.txt", "w")
        file.writelines(installs)
        file.close()

def check_if_errors(output):
    return 'error' in output.keys()


def debug_errors(code, llm=llm_palm2):
    """
    Create a Chat Template to handle errors and exceptions.
    
    Process
    - Gather the output of running the code
    - Generate a prompt to solve any errors from the saved output

    Args:
    - input (str): The LLM-generated code.
    """
    
    output = execute_code(code)
    print("Output\n", output)
    # Iterate 5 times to try to get rid of bugs. If bugs cannot be removed after 5 iterations, time out
    if check_if_errors(output):
        for i in range(0,5):
                # Define the prompt for invocation
                template_error = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful coding assistant. You have been given the following code and error outputs. Output a modified version of the code that doesn't contain any errors."),
                    ("user", "Code: {code}\n Error Output: {error}")
                ])

                debug_chain = template_error | llm | output_parser

                code_output = debug_chain.invoke({'code': code, 'error': output.get('error')})

                # Gather the new code from the template
                new_code = re.search(r'```python(?s:(.*?))```', code_output).group(1)

                # Execute the code and save the output
                output = execute_code(new_code)
                print("Output\n", output)
                if not check_if_errors(output):
                    # No errors so we return
                    return
    

def gen_code(input, files, llm_prompt=llm_mixtral22, llm_code=llm_palm2):
    """
    High level function to generate python code from a user prompt and file.

    Process
    - Generate an expanded prompt from the user prompt
    - Generate python code from the expanded prompt

    Args:
    - input (str): The user prompt.
    - files (str): The path to the file.

    """
    requirements_file = open("requirements.txt", 'r')

    contents = requirements_file.read()

    file_contents = get_file_contents(files)

    prompt_chain = create_prompt_template() | llm_prompt | output_parser
    
    code_chain = create_code_template() | llm_code | output_parser
    
    prompt_output = prompt_chain.invoke({"input": input, "file": file, "contents": file_contents})
    
    code_output = code_chain.invoke({"input": prompt_output, "file": file, "contents": file_contents, "requirements": contents})
    
    python_code = re.search(r'```python(?s:(.*?))```', code_output).group(1)

    return python_code




if __name__ == "__main__":
    #input = "change this into picture.dcm and save it in the same directory as the inputted file"
    
    input = "change this into picture.jpg"
    file = "images/picture.png"

    python_code = gen_code(input, file)
    install_requirements(python_code)
    print(python_code)
    print(debug_errors(python_code))