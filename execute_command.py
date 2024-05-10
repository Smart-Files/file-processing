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


print(execute_command("ls -l"))