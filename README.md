# SmartFile File Processing



## Setting up Developer Environment

### Using **pipenv**

> Since this project will later be deployed on Docker, we use **pipenv** to virtualize and manage dependencies to prevent later conflict. All development and testing should be carried out in the **pipenv** virtual environment.


Installing **pipenv** globally (Python installation required)
```sh
pip install pipenv --user
```


Installing packages from existing project with `requirements.txt`:
```sh
pipenv install
```


Activating the virtual environment
```sh
pipenv shell
```

Installing specific packages
```sh
pipenv install <package>
```


It's also possible to update out-of-date packages
```sh
pipenv update
```

### Setting up environment variables

> Since we are using the openrouter.ai API for testing our system, we will need to use an API key. We will use a .env file to store our API key. **pipenv** will automatically import variables from `.env` into the environment variables.

Create a file named `.env` in the main repository directory and add the following
```sh
OPENROUTER_API_KEY=your-api-key
```

Using the API key within a Python program
```py
import os
API_KEY = os.getenv('OPENROUTER_API_KEY')
print(API_KEY) # your-api-key
```