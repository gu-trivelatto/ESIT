# Energy System Insight Tool (ESIT)
This is a LLM-based tool to help people interested in understanding the impacts of politics taken torwards energy systems modelling without requiring any previous technical knowledge. The tool is able to understand the scenarios proposed textually by the user and, based on that,
apply modifications to the model, run it, compare the results and show the consequences of the changes both in textual and graphical forms.
It can also provide precise information about the used base model, as well as it's goals and theoric details.


## Cloning the repository

Since there is a submodule included in this project, the cloning instructions have some more steps than normal.

If you are cloning this repository for the first time, you can run the following:

```console
> git clone --recurse-submodules https://github.com/gu-trivelatto/ESIT.git
```

This will ensure that the submodules are initialized and updated.

If you cloned this repository without initializing submodules, you can do the following:

```console
> git submodule update --init --recursive
```

This has the same effect as running `git clone` with the submodule setting.

## Installing the dependencies

1. Change the directory to the repository:
    ```console
    > cd ESIT
    ```
2. Install the package:
    ```console
    > pip install -e .
    ```

## Setting up the environment

1. Copy the `secrets_template.yml` file and rename it as `secrets.yml`:
   ```console
   > cd metadata
   > cp secrets_template.yml secrets.yml
   ```
2. Replace the content of the fields with your keys for the necessary services

## Running the application

1. Enter the root folder of the project
2. Run the app:
   ```console
   > python esit.py
   ```

If you want to activate the debugging mode, run it as:

```console
> python esit.py -d
```
