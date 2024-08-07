## Cloning the repository

Since there is a submodule included in this project, the cloning instructions have some more steps than normal.

If you are cloning this repository for the first time, you can run the following:

	```console
	> git clone --recurse-submodules https://github.com/gu-trivelatto/master_thesis.git
	```
This will ensure that the submodules are initialized and updated.

If you cloned this repository without initializing submodules, you can do the following:

    ```console
    > git submodule update --init --recursive
    ```
This has the same effect as running `git clone` with the submodule setting.

## Installing the dependencies

1. Change the directlry to the repository:

    ```console
    > cd ESIT
    ```
2. Install the package:

    ```console
    > pip install -e .
    ```

# Outdated

## Install LangChain
pip install langchain

## Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

## Install Ollama for Python
pip install ollama
