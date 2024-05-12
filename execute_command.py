import subprocess
import shlex
import os
import re
from pathlib import Path
from functools import partial

def get_current_directory_contents():
    current_directory = os.getcwd()
    contents = os.listdir(current_directory)
    return contents

def execute_command_factory(uuid: str):
    return partial(execute_command, uuid)

def execute_command(uuid: str, command: str) -> dict["stdout": str, "stderr": str]:
    """Executes a command in the shell and returns the output.

    Args:
    - command: str - The command to execute in the shell.

    Returns:
    - output: str - The output of the command.
    """

    command = command.strip()
    if command[0] in ["'", "`", '"', '`'] and command[len(command)-1] == command[0]:
        command = command[1:len(command)-1]

    # Change the current working directory to the specified directory

    cwd = os.getcwd()
    print(cwd)
    Path(f"/app/working_dir/{uuid}").mkdir(parents=True, exist_ok=True)
    os.chdir(f"/app/working_dir/{uuid}")

     
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # Capture stderr as well
        text=True,  # Decode the output to strings
        shell=True
    )

    print("Command execution completed.")
    print("Return Code:", result.returncode)
    print("Standard Output:", result.stdout)
    print("Standard Error:", result.stderr)

    print("RESULT: ", result)

    stdout = None
    if (result.stdout):
        stdout = result.stdout

    stderr = None
    if (result.stderr):
        stderr = result.stderr

    output = f"""Command: {command}

    STDOUT: 
    {stdout}

    STDERR:
    {stderr}

    Return Code: {result.returncode}

    Current Directory Contents:
    {get_current_directory_contents()}
    """


    os.chdir(cwd)
    print(output)
    return output
