from setuptools import setup, find_packages

setup(
    name='esit',
    version='1.0.0',
    packages=find_packages(),
    description='Energy System Insight Tool (ESIT)',
    long_description="""
Energy System Insight Tool (ESIT)

This is a LLM-based tool that is capable of talking to the user about any Energy System Model
modeled in a format compatible with the Compact Energy System Modeling Tool (CESM), which is a
submodule of this project (also available in https://github.com/EINS-TUDa/CESM).

This tool is able of understanding the parameters and energy systems defined in the model, as
well as modifying them, running the new model, comparing results with the original model and
plotting the available results. The tool is also capable of searching for information online
to complement any locally available information.

The German energy system model for which this tool was designed is based on the results of the
paper "Barbosa, Julia, Christopher Ripp, and Florian Steinke. Accessible Modeling of the German
Energy Transition: An Open, Compact, and Validated Model. Energies 14, no. 23 (2021)", and although
the tool is able to answer about any energy system model (given that the format is compatible with
CESM), it will be more reliable in the presence of a supporting paper, which is the case of the
cited model.
    """,
    long_description_content_type='text/markdown',
    py_modules=['esit'],
    entry_points={
        'console_scripts': [
            'esit = esit:app',
        ],
    },
    install_requires=[
        "setuptools",
        "openpyxl",
        "numpy",
        "scipy",
        "pandas",
        "gurobipy",
        "plotly",
        "kaleido",
        "click",
        "InquirerPy",
        "SQLAlchemy",
        "pyarrow",
        "customtkinter",
        "tkinter",
        "langgraph",
        "langchain",
        "langchain_core",
        "langchain_groq",
        "langchain_community",
        "langchain_text_splitters",
        "openpyxl"
    ],
    # Metadata
    author=['Gustavo Trivelatto Gabriel'],
    author_email='gu.trivelatto@yahoo.com.br',
    # TODO update the URL if changing the name of the repo
    url='https://github.com/gu-trivelatto/master_thesis',
    license='MIT',
)