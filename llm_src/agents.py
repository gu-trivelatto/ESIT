import os
import math
import sqlite3
import subprocess
import numpy as np
import pandas as pd
import customtkinter
import tkinter as tk

from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from CESM.core.model import Model
from CESM.core.input_parser import Parser
from openpyxl import load_workbook
from abc import ABC, abstractmethod
from llm_src.state import GraphState
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

# TODO standardize variable names used from the state
# TODO standardize the way the agents interact with the state

class HelperFunctions(ABC):
    def __init__(self):
        pass

    def get_params_and_cs_list(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")

        conversion_processes = np.asarray(df.iloc[:,0].dropna())
        mask = np.where(conversion_processes != 'DEBUG')
        conversion_processes = conversion_processes[mask]
        parameters = np.asarray(df.columns[4:])
        descriptions = np.asarray(df.iloc[0,4:])
        param_n_desc = np.empty((len(parameters),1),dtype=object)
        
        for i in range(len(parameters)):
            param_n_desc[i] = f'{parameters[i]} - {descriptions[i]}'

        cs = np.asarray(df.iloc[:,0:3].dropna())
        mask = np.where(cs[:,0] != 'DEBUG')
        cs = cs[mask]
        conversion_subprocesses = np.empty((len(cs),1),dtype=object)

        for i in range(len(cs)):
            conversion_subprocesses[i] = f'{cs[i,0]}@{cs[i,1]}@{cs[i,2]}'

        return param_n_desc, conversion_subprocesses

    def get_scenario_params(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"Scenario")

        base_index = df.index[df['scenario_name'] == 'Base'].tolist()[0]
        discount_rate = df['discount_rate'][base_index]
        annual_co2_limit = df['annual_co2_limit'][base_index]
        co2_price = df['co2_price'][base_index]
        
        if math.isnan(discount_rate):
            discount_rate = None
        if math.isnan(co2_price):
            co2_price = None

        return {'discount_rate': discount_rate, 'annual_co2_limit': annual_co2_limit, 'co2_price': co2_price}

    def get_conversion_processes(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"Commodity")

        cond = df['commodity_name'] != 'Dummy'
        cond = cond & (df['commodity_name'] != 'DEBUG')
        cond = cond & (df['commodity_name'].str.contains('Help') == False)

        return df['commodity_name'][cond].tolist()
    
    def get_yearly_variations(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")

        results = []
        for col in df.columns[10:-2]:
            cond = (~df[col].isna()) & (df[col].str.contains(';'))
            values = df[col][cond].tolist()
            CPs = df['conversion_process_name'][cond].tolist()
            cin = df['commodity_in'][cond].tolist()
            cout = df['commodity_out'][cond].tolist()
            for i in range(len(values)):
                results = results + [[f'{CPs[i]}@{cin[i]}@{cout[i]}', values[i]]]
            
        return results

    def get_cs_param_selection(self, techmap_file, cs_list, param_list):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")
        result = []
        
        for cs in cs_list:
            split_cs = cs.split('@')

            if len(split_cs) > 1:
                cond = df['conversion_process_name'] == split_cs[0]
                cond = cond & (df['commodity_in'] == split_cs[1])
                cond = cond & (df['commodity_out'] == split_cs[2])
            else:
                cond = df['conversion_process_name'] == split_cs[0]
            
            for param in param_list:
                try:
                    if len(split_cs) > 1:
                        result = result + [cs, param, df[param][cond].values[0], df[param][1]]
                    else:
                        for i in range(sum(cond)):
                            idx = df[param][cond].index[i]
                            cs = f'{split_cs[0]}@{df["commodity_in"][idx]}@{df["commodity_out"][idx]}'
                            result = result + [cs, param, df[param][cond].values[i], df[param][1]]
                except:
                    pass

        return result

    def consult_info(self, query, techmap_file):
        consult_type = query['consult_type']
        
        if consult_type == 'yearly_variation':
            info = self.get_yearly_variations(techmap_file)
        elif consult_type == 'cs_param_selection':
            info = self.get_cs_param_selection(techmap_file, query['cs'], query['param'])
        else:
            info = 'Consult type not recognized'
        
        return info

    def modify_model(self):
        pass

class InputGetter(ABC):
    def __init__(self, *args, **kwargs):
        # Extract label, buttons, and callback from kwargs
        label = kwargs.pop('label', None)
        buttons = kwargs.pop('buttons', [])
        app = kwargs.pop('app', None)
        self.callback = kwargs.pop('callback', None)

        #super().__init__(*args, **kwargs)
        self.toplevel = customtkinter.CTkToplevel(app)
        self.toplevel.geometry("800x600")

        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(self.toplevel, borderwidth=0, background="#ffffff")
        self.scrollbar = customtkinter.CTkScrollbar(self.toplevel, orientation="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = customtkinter.CTkFrame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=800)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse scroll events
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_scroll)
        self.canvas.bind_all("<Button-4>", self._on_mouse_scroll)
        self.canvas.bind_all("<Button-5>", self._on_mouse_scroll)

        # Configure the grid layout for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Add widgets to the scrollable frame
        if label:
            self.label = customtkinter.CTkLabel(self.scrollable_frame, text=label)
            self.label.grid(row=0, column=0, padx=10, pady=10)
        
        if buttons:
            for i in range(len(buttons)):
                button = customtkinter.CTkButton(self.scrollable_frame, text=buttons[i], command=lambda b=buttons[i]: self.button_clicked(b))
                button.grid(row=i+1, column=0, padx=10, pady=10)

    def _on_mouse_scroll(self, event):
        if event.delta:
            self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def button_clicked(self, text) -> None:
        if self.callback:
            self.callback(text)
        self.destroy()

class AgentBase(ABC):
    def __init__(self, chat_model, json_model, base_model, mod_model, state: GraphState, app, debug):
        self.chat_model = chat_model
        self.json_model = json_model
        self.state = state
        self.debug = debug
        self.app = app
        self.selected_value = None
        self.base_model = base_model
        self.mod_model = mod_model
        
    def confirm_selection(self, selected_value):
        self.selected_value = selected_value

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        pass

    def execute(self) -> GraphState:
        pass
    
class ResearchAgentBase(ABC):
    def __init__(self, chat_model, json_model, retriever, web_tool, state: GraphState, app, debug):
        self.retriever = retriever
        self.web_tool = web_tool
        self.chat_model = chat_model
        self.json_model = json_model
        self.state = state
        self.debug = debug
        self.app = app
        self.selected_value = None
        
    def get_answer_analyzer_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at summarizing a bunch of data to extract only the important bits from it.

            Given the user's QUERY and the SEARCH_RESULTS, summarize as briefly as possible the information
            searched by the user. Don't give any preamble or introduction, go directly to the summary
            of the requested information.
            
            If it helps to provide a more precise answer, you can also make use of the CONTEXT.
            
            Whenever there is a source in the data received in SEARCH_RESULTS, you MUST include the used
            sources as a topic at the end of the summarization. In case there is no source, simply ignore this
            instruction.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            SEARCH_RESULTS: {search_results} \n
            CONTEXT: {context} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","search_results","context"],
        )
    
    def confirm_selection(self, selected_value):
        self.selected_value = selected_value

    @abstractmethod
    def get_first_prompt_template(self) -> PromptTemplate:
        pass
    
    def get_second_prompt_template(self) -> PromptTemplate:
        pass

    def execute(self) -> GraphState:
        pass

class DateGetter(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        num_steps = self.state['num_steps']
        num_steps += 1
        
        current_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        result = f'The current date and time are {current_date}'
        
        if self.debug:
            print("---DATE GETTER TOOL---")
            print(f'CURRENT DATE: {current_date}\n')

        self.state['context'] = [result]
        self.state['num_steps'] = num_steps

        return self.state

class TypeIdentifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of the Energy System Insight Tool (ESIT), your main task is to
            initiate the pipeline of the tool by deciding which type of request the user
            is trying to ask. You have three execution branches available, "energy_system",
            "mixed" and "general". \n

            "energy_system": the USER_INPUT is related to modeling. Unless the user specifies that the
            modeling he is talking about is not related to the model this tool is projected to
            analyze, you will use this branch. This branch allows you to complete diverse tasks
            related to Energy System modeling, like consulting information about the model,
            modifying values of the model, running it, comparing and plotting results, and also
            consults to the paper related to the model for more details. Anything on these lines
            should use this branch. \n
            
            "mixed": the USER_INPUT would be identified as "energy_system", but there are references in
            the input that must be researched online to be able to gather the necessary context for the
            modelling tools to reach the user's goals. \n
            
            "general": the USER_INPUT is related to some generic topic, it may consist of one or more
            points that require searching for information. \n
            
            You must output a JSON with a single key 'input_type' containing exclusivelly the 
            selected type. \n
            
            You may also use the CHAT_HISTORY to get a better context of past messages exchanged
            between you and the user, and better understand what he wants in the current message. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","history"],
        )
    
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"user_input": user_input, "history": history})
        selected_type = llm_output['input_type']
        
        if self.debug:
            print("---TYPE IDENTIFIER---")
            print(f'USER INPUT: {user_input.rstrip()}')
            print(f'IDENTIFIED TYPE: {selected_type}\n')
            print(history)
        
        self.state['input_type'] = selected_type
        self.state['num_steps'] = num_steps
        
        return self.state

class ESActionsAnalyzer(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at verifying the provided CONTEXT and ACTION_HISTORY of a model
            interaction to decide what is the next step to be taken based on the USER_INPUT.
            You know how to avoid repeating instructions and do only the necessary to reach
            the user's request. \n
            
            You have a total of 6 actions that can be taken. You can modify the model, run the model,
            consult information from the model, compare the results of the model, plot the results
            of the model and no action in case there is nothing more to be done and the output can
            be generated to the user. \n
            
            The user will provide you with a USER_INPUT and you should define the line of action to take.
            CONTEXT and ACTION_HISTORY is also there for you to know what you already did in past iterations. \n
            
            You must output a JSON object with a single key called 'action', this key can contain one
            of the following values: ['modify', 'run', 'consult', 'compare', 'plot', 'no_action']. \n
            
            Running a model is expensive, so, unless the user asks you explicitly to run the model in
            a certain way, or he asks for the differences in results after changing certain aspects of
            the model, don't run it. \n
            
            You may also check for the CHAT_HISTORY to verify what already happened in this session
            of the chat with the user. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            ACTION_HISTORY: {action_history} \n
            CONTEXT: {context} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input", "action_history", "context", "history"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        action_history = self.state['action_history']
        context = self.state['context']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "action_history": action_history, "context": context, "history": history})
        selected_action = llm_output['action']
        
        if self.debug:
            print("---ENERGY SYSTEM ACTIONS---")
            print(f'USER INPUT: {user_input.rstrip()}')
            print(f'SELECTED ACTION: {selected_action}\n')
            
        self.state['next_action'] = selected_action
        self.state['num_steps'] = num_steps
        
        return self.state

class ContextAnalyzer(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at deciding if the already available information is enough to
            fully answer the user's query. If not you must try to gather the missing info. \n
            
            Given a USER_INPUT and the available CONTEXT, decide if the available information
            is already enough to answer the query proposed by the user. You have also access to
            an array QUERY_HISTORY which contains all the past queries you generated, so you can avoid
            repetitions if the information was not found previously. \n
            
            You may also check for the CHAT_HISTORY to verify what already happened in this session
            of the chat with the user. \n
            
            The pipeline you're working with has some tools that can help find information, your
            main purpose is to decide whether or not the current information in CONTEXT can fullfil
            the query asked by the user. \n
            
            Your output should be a JSON object containing two keys, 'ready_to_answer' and
            'next_query'. 'ready_to_answer' is a boolean that indicates if all necessary info is
            present and 'next_query' is the query you'll develop for the info gathering tool
            to search for the missing information. \n
            
            Never ask anything to the user, the only kind information you should try to gather is
            related to details of the query that don't need interaction with the user to be
            answered, such as information that can be found in specific documents, on the internet
            or simply by directly answering the user. \n
            
            NEVER ASK FOR MORE CONTEXT TO THE USER, THE ONLY INFORMATION YOU ARE ALLOWED
            TO REQUEST IS INFORMATION MISSING FOR THE ANSWER, NOT FROM THE QUESTION. THE USER
            MAY ASK SOME GENERIC QUERIES, JUST ANSWER THEM, DON'T ASK FOR MORE CONTEXT ON
            DETAILS OF WHAT THE USER WANTS. IF THE QUERY IS GENERIC, GATHER ALL THE INFORMATION
            YOU CAN WITHOUT ASKING THE USER, AND ANSWER WITH WHAT YOU CAN. \n
            
            Also, never consider that you know how to to calculations, if there is any
            calculation in the user request, build a query with it and pass it forward. \n
            
            In the following situations you must output 'next_query' as "<KEEP_QUERY>":
            - User asks to modify parameters or characteristics of an energy system model;
            - Plotting, they don't require extra information, the tools can handle it perfectly;
            - User asks you to run a new simulation on an energy modeling system;
            - User gives you a direct command related to modelling;
            - The user asks anything about LangSmith (understand that as having the word LangSmith) \n
            
            Consider that for you boolean answer the words false and true should always be written
            in full lower case. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            QUERY_HISTORY: {query_history} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","query_history","history"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        ## Get the state
        user_input = self.state['user_input']
        context = self.state['context']
        query_history = self.state['query_history']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input,
                                   "context": context,
                                   "query_history": query_history,
                                   "history": history
                                   })
        
        if llm_output['next_query'] == '<KEEP_QUERY>':
            llm_output['next_query'] = self.state['user_input']
        
        if self.debug:
            print("---CONTEXT ANALYZER---")
            print(f'NEXT QUERY: {llm_output}\n')
            print(history)
        
        self.state['query_history'] = query_history + [llm_output['next_query']]
        self.state['next_query'] = llm_output
        self.state['num_steps'] = num_steps
        
        return self.state

class Mixed(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at reading the user USER_INPUT and the available CONTEXT to decide if there
            is already enough information gathered to fulfill the energy system related command
            made by the user. \n
            
            You must be certain that you have all the data before deciding to send it to the
            modelling section of the pipeline. If any of the values asked by the user is not
            directly given by him, you can't consider the data complete unless you have the
            desired value in the CONTEXT. \n
            
            You may also check for the CHAT_HISTORY to verify what already happened in this session
            of the chat with the user. \n

            You must output a JSON object with a single key 'is_data_complete' containing a boolean
            on whether you have enough data for the user's request or not. \n
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            CONTEXT: {context} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","history"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        context = self.state['context']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "context": context, "history": history})
        is_data_complete = llm_output['is_data_complete']
        
        if self.debug:
            print("---TOOL SELECTION---")
            print(f'USER INPUT: {user_input.rstrip()}')
            print(f'CONTEXT: {context}')
            print(f'DATA IS COMPLETE: {is_data_complete}\n')
            
        self.state['is_data_complete'] = is_data_complete
        self.state['num_steps'] = num_steps
        
        return self.state

# TODO update this selector, the query is kinda strange now

class ToolSelector(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at reading a QUERY generated by another agent in this system and routing to
            our internal knowledge system or directly to final answer. \n

            Use the following criteria to decide how to route the query to one of our available tools: \n\n
            
            If the user asks anything about LangSmith, you should use the 'RAG_retriever' tool.
            
            For any mathematical problem you should use 'calculator'. Be sure that you have all the necessary
            data before routing to this tool. Also, it only solves the following calculations [+,-,*,/,^], it
            doesn't do anything else.

            If you are unsure or the person is asking a question you don't understand then choose 'web_search'.
            
            If the query seems like a question that should be asked to the user, use 'user_input'. However, this
            should be used only in a last case scenario, since clarifications to the user should be rare. If
            you think the question can be found online or in the documents we have, use them instead.

            You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web_search.
            Give a choice contained in ['RAG_retriever','calculator','web_search'].
            Return the a JSON with a single key 'router_decision' and no premable or explaination.
            Use the initial query of the user and any available context to make your decision about the tool to be used.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY : {query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        query = self.state['next_query']['next_query']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"query": query})
        router_decision = llm_output['router_decision']
        
        if self.debug:
            print("---TOOL SELECTION---")
            print(f'QUERY: {query}')
            print(f'SELECTED TOOL: {router_decision}\n')
        
        self.state['selected_tool'] = router_decision
        self.state['num_steps'] = num_steps
        
        return self.state
    
# TODO rename prompt functions
# TODO rewrite this and re-purpose as the paper RAG, move to the other branch
# TODO check the answer analyzer prompt

class ResearchInfoRAG(ResearchAgentBase):
    def get_first_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a master at working out the best questions to ask our knowledge agent to get 
            the best info for the customer. \n

            Given the QUERY, work out the best questions that will find the best info for 
            helping to write the final answer. Write the questions to our knowledge system not 
            to the customer. \n

            Return a JSON with a single key 'questions' with no more than 3 strings of and no 
            preamble or explaination. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
        
    def get_second_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an assistant for question-answering tasks. Use the following pieces of 
            retrieved context to answer the question. If you don't know the answer, just say 
            that you don't know. Use three sentences maximum and keep the answer concise. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUESTION: {question} \n
            CONTEXT: {context} \n
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question","context"],
        )
        
    def execute(self) -> GraphState:
        question_rag_prompt = self.get_first_prompt_template()
        rag_prompt = self.get_second_prompt_template()
        answer_analyzer_prompt = self.get_answer_analyzer_prompt_template()
        
        question_rag_chain = question_rag_prompt | self.json_model | JsonOutputParser()
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | self.chat_model
            | StrOutputParser()
        )
        answer_analyzer_chain = answer_analyzer_prompt | self.chat_model | StrOutputParser()
        
        if self.debug:
            print("---RAG LANGSMITH RETRIEVER---")
            
        query = self.state['next_query']['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1

        questions = question_rag_chain.invoke({"query": query})
        questions = questions['questions']

        rag_results = []
        for idx, question in enumerate(questions):
            temp_docs = rag_chain.invoke(question)
            if self.debug:
                print(f'QUESTION {idx}: {question}')
                print(f'ANSWER FOR QUESTION {idx}: {temp_docs}')
            question_results = question + '\n\n' + temp_docs + "\n\n\n"
            if rag_results is not None:
                rag_results.append(question_results)
            else:
                rag_results = [question_results]
        if self.debug:
            print(f'FULL ANSWERS: {rag_results}\n')
        
        processed_searches = answer_analyzer_chain.invoke({"query": query, "search_results": rag_results, "context": context})
        result = f'Source: Langsmith Doc \n{query}: \n{processed_searches}'
        
        # TODO keeping the questions is not necessary
        
        self.state['context'] = context + [result]
        self.state['num_steps'] = num_steps
        
        return self.state

class ResearchInfoWeb(ResearchAgentBase):
    def get_first_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a master at working out the best keywords to search for in a web search to get the best info for the user.

            Given the QUERY, work out the best search queries that will find the info requested by the user
            The queries must contain no more than a phrase, since it will be used for online search. 

            Return a JSON with a single key 'keywords' with at most 3 different search queries.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
        
    def execute(self) -> GraphState:
        if self.debug:
            print("---RESEARCH INFO SEARCHING---")
            
        query = self.state['next_query']['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        prompt = self.get_first_prompt_template()
        answer_analyzer_prompt = self.get_answer_analyzer_prompt_template()
        
        llm_chain = prompt | self.json_model | JsonOutputParser()
        answer_analyzer_chain = answer_analyzer_prompt | self.chat_model | StrOutputParser()

        # Web search
        keywords = llm_chain.invoke({"query": query, "context": context})
        keywords = keywords['keywords']
        full_searches = []
        for idx, keyword in enumerate(keywords):
            temp_docs = self.web_tool.execute(keyword)
            if type(temp_docs) == list:
                for d in temp_docs:
                    web_results = f'Source: {d["url"]}\n{d["content"]}\n'
                web_results = Document(page_content=web_results)
            elif type(temp_docs) == dict:
                web_results = f'\nSource: {d["url"]}\n{d["content"]}'
                web_results = Document(page_content=web_results)
            else:
                web_results = 'No results'
            if self.debug:
                print(f'KEYWORD {idx}: {keyword}')
                print(f'RESULTS FOR KEYWORD {idx}: {web_results}')
            if full_searches is not None:
                full_searches.append(web_results)
            else:
                full_searches = [web_results]

        processed_searches = answer_analyzer_chain.invoke({"query": query, "search_results": full_searches, "context": context})
        
        if self.debug:
            print(f'FULL RESULTS: {full_searches}\n')
            print(f'PROCESSED RESULT: {processed_searches}\n')
        
        self.state['context'] = context + [processed_searches]
        self.state['num_steps'] = num_steps
        
        return self.state
    
class Calculator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an specialist at identifying the correct operation and operands to build
            a JSON object as an instruction to a calculator tool. You will identify the correct
            parameters from the given QUERY. \n
            
            You must output a JSON object with three keys are 'operation', 'op_1' and 'op_2', where
            'operation' is the operation to be executed, and 'op_1' and 'op_2' are the operands of
            the specific operation. \n
            
            'operation' can only be [+,-,*,/,^], and 'op_1' and 'op_2' must be integers or float. \n
            
            If you judge that the equation consists of multiple operations, you can check the CONTEXT
            to verify if the results of some of the operations is already known, and then you may use
            it to complete your selected operation. If not, simply choose the first operation in the
            correct order to be executed. The result will be saved and used for the next operation
            in another iteration. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            CONTEXT: {context}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","context"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
    
        query = self.state['next_query']['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"query": query, "context": context})
        
        operation = llm_output['operation']
        op_1 = llm_output['op_1']
        op_2 = llm_output['op_2']

        if operation == "+":
            result = op_1 + op_2
        elif operation == "-":
            result = op_1 - op_2
        elif operation == "/":
            result = op_1 / op_2
        elif operation == "*":
            result = op_1 * op_2
        elif operation == "^":
            result = op_1 ** op_2
        else:
            result = 'ERROR'
            
        if result == 'ERROR':
            str_result = f'The selected operation "{operation}" was not recognized.'
        else:
            str_result = f'{op_1} {operation} {op_2} = {result}'
        
        if self.debug:
            print("---CALCULATOR TOOL---")
            print(f'OPERATION: {operation}')
            print(f'OPERAND 1: {op_1}')
            print(f'OPERAND 2: {op_2}')
            print(f'RESULT: {str_result}\n')
            
        self.state['context'] = context + [str_result]
        self.state['num_steps'] = num_steps
        
        return self.state

# TODO review the usage of 'final_answer'

class RunModel(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        action_history = self.state['action_history']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        if self.debug:
            print('---SIMULATION RUNNER---')
        
        try:
            model_name = self.mod_model.split('/')[-1]
            model_name = model_name[:-5]
            scenario = 'Base'
            techmap_dir_path = Path("../CESM").joinpath('Data', 'Techmap')
            ts_dir_path = Path("../CESM").joinpath('Data', 'TimeSeries')
            runs_dir_path = Path("../CESM").joinpath('Runs')
            
            # Create a directory for the model if it does not exist
            db_dir_path = runs_dir_path.joinpath(model_name+'-'+scenario)
            if not runs_dir_path.exists():
                os.mkdir(runs_dir_path)
            
            if not os.path.exists(db_dir_path):
                os.mkdir(db_dir_path)

            # Create and Run the model
            conn = sqlite3.connect(":memory:")
            parser = Parser(model_name, techmap_dir_path=techmap_dir_path, ts_dir_path=ts_dir_path, db_conn = conn, scenario = scenario)

            # Parse
            if self.debug:
                print("\n#-- Parsing started --#")
            parser.parse()

            # Build
            if self.debug:
                print("\n#-- Building model started --#")
            model_instance = Model(conn=conn)

            # Solve
            if self.debug:
                print("\n#-- Solving model started --#")
            model_instance.solve()

            # Save
            if self.debug:
                print("\n#-- Saving model started --#")
            model_instance.save_output()

            
            db_path = db_dir_path.joinpath('db.sqlite')
            if db_path.exists():
                # Delete the file using unlink()
                db_path.unlink()
            
            # write the in-memory db to disk
            disk_db_conn = sqlite3.connect(db_path)
            conn.backup(disk_db_conn)
            result = 'The model has runned successfully!'
        except:
            result = 'Something went wrong and the model has not completely runned.'
            
        self.state['num_steps'] = num_steps
        self.state['action_history'] = action_history + [result]
        
        return self.state

# TODO separate the model modification steps into functions of the Helper

class ModifyModel(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def get_sheet_selector_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at identifying the correct sheet to be modified based on the user's USER_INPUT. \n
            
            You must output a JSON object with a single key 'sheet', that should receive 'conversionsubprocess'.
            You should only output 'scenario' instead if the user explicitly talks about co2 limit, co2 price
            or discount rate, these terms must be included in the query. \n
            
            The only exception to this rule is that if the USER_INPUT leads to the selection to be 'scenario' but
            SCENARIO_MODDED is True, then you should output 'conversionsubprocess' anyway. \n
            
            Remember that for the sheet to be 'scenario' you need to receive explicitly the defined terms from
            the USER_INPUT. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            SCENARIO_MODDED: {scen_modded}
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["user_input", "scen_ready"],
        )
    
    def get_scenario_param_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at modifying the scenario of a model based on the user's USER_INPUT. \n
            
            There are three parameters you can change: discount_rate, annual_co2_limit and co2_price.
            In SCEN_PARAMS you have a dictionary with the current values for each of them, and you should
            decide how to change them to fulfill the user's request. \n
            
            You must output a JSON object with a single element 'new_values' that contains the modified
            values for the three in a three elements list. If you haven't modified some of them, simply
            output the same value you received. The order should be the same as the input.\n
            
            For the anual co2 limit, if the user request a change after a specific year you should always
            add the next year to the yearly list as a starting point for the modification, and keep the remaining
            years, just changing their values. Also, for this list the only possibility are numbers for the 
            values, not 'infinity' or anything like that. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            SCEN_PARAMS: {scen_params}
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["user_input", "scen_params"],
        )

    def get_params_general_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at identifying the correct way to modify a model based on the user's USER_INPUT. \n
            
            You are part of a tool where there are two other agents ready to execute their specific tasks
            related to model modification. Your goal is to identify which one to use depending on the user's
            input. \n
            
            The two possibilities are:
            1. The user provides one or more specific instructions of modifications that should be done to the model,
            with the selection of specific values for each modification;
            2. The user asks for a specific scenario where the tool should decide which parameters and subprocesses
            should be modified, as well as deciding the correct value to change the combinations to. \n
            
            Your output must be a JSON object with a single key called 'parametrization_type', where your options are
            'defined' for the first case and 'undefined' for the second. These are your only possibilities. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["user_input"],
        )

    def get_params_defined_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at identifying the parameters the user wish to modify in the model, as well as the 
            conversion subprocesses and new values. \n
            
            As a context, you will receive two data arrays. PARAMS provides you the name of the parameters
            available to be selected formated as a combination of the actual parameter name and their description
            in the format 'name - description'. CONVERSION_SUBPROCESSES provides you the combination of 'cp'
            (conversion process name), 'cin' (commodity in), 'cout' (commodity out)  in the format 'cp@cin@cout'. \n
            
            The user's USER_INPUT may contain a request to change one or more combinations of parameter and 
            conversion subprocess. \n
            
            Your goal is to output a JSON object containing a single key 'param_cs_selection', which should contain
            a list. \n
            
            The composition of the list is the following: For each combination of conversion subprocess, parameter and
            value asked by the user there should be a list with four elements in 'params_cs_selection'. Each sub list
            is composed of ['parameter', ['cs_match'], 'new_value', 'mod_type']. The output structure you should
            always follow is [[combination], [combination], [combination], ...], and this list that comprehends
            all of the combinations of parameter, cs, value and modification type is the single element of 'params_cs_selection'.\n
            
            NEVER MAKE UP DATA, USE ONLY DATA FROM THE GIVEN LISTS. NEVER MODIFY THE SELECTED ENTRY, USE IT AS YOU
            FOUND IT IN THE LIST! THE COMBINATION 'cp@cin@cout' MUST MATCH EXACTLY WITH THE ENTRIES OF THE LIST.
            YOU CAN NEVER GET cp, cin OR cout FROM DIFFERENT ENTRIES, THEY MUST ALWAYS COME FROM THE SAME. \n
            
            For 'parameter', the value is a string and you must always output either one selected param or 'NOT_FOUND'
            if you couldn't match any of the available ones (never output more than one per combination). \n
            
            For 'cs_match' you should always output a list with the conversion suprocesses matches (in format cp@cin@cout).
            If there is a sigle matching conversion subprocess, then the list will have a single element but will still 
            be a list. And if you can't match any conversion subprocess, then you must output an empty list. Important that 
            there is a difference between asking for a generic conversion process with more than a single conversion 
            subprocess related to it (situation in which you should match all options to the combination) and asking to
            change a set of conversion subprocesses with the same conversion process name (situation in which each of
            the conversion subprocesses should be given their own combinations) \n
            
            For the 'new_value' you have four possibilities:
            1. The user specified a single value for the combination. In this case you put the value as a numeric value
            in the combination list;
            2. The user specified a value containing [] in it. In this case you should use the exact same value as a string.
            DON'T CHANGE ANYTHING IN THE VALUE. For example, the user inputs [2020 10; 2030 5; 2040 2], you should output
            exactly [2020 10; 2030 5; 2040 2] as a string for the value of this combination;
            3. There is a indirect reference to the desired value, in this case the value will probably be in the provided
            CONTEXT, check there before going to the fourth possibility;
            4. If there is no value to be found for a specific combination, simply output 'NOT_PROVIDED' for that one,
            NEVER MAKE UP VALUES. 'NOT_PROVIDED' is the only possible text that you can output as a value.\n
            
            For the 'mod_type' you can use two different types, 'fixed' or 'multiplier'. 'fixed' tells the tool
            that you want the value to be exactly what you specified, 'multiplier' indicates that you want the
            current value to be multiplied by the value you defined. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            PARAMS: {params} \n
            CONVERSION_SUBPROCESSES: {CSs} \n
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["user_input","context","params","CSs"],
        )

    def get_params_undefined_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at deciding the necessary modifications in the model based on the user's USER_INPUT. \n
            
            As a context, you will receive two data arrays. PARAMS provides you the name of the parameters
            available to be selected formated as a combination of the actual parameter name and their description
            in the format 'name - description'. CONVERSION_SUBPROCESSES provides you the combination of 'cp'
            (conversion process name), 'cin' (commodity in), 'cout' (commodity out)  in the format 'cp@cin@cout'. \n
            
            Your goal is to output a JSON object containing a single key 'param_cs_selection', which should contain
            a list. This list contains each combination of conversion subprocess, parameters, values and modification
            type that you judge necessary to fulfill the user's scenario. \n
            
            For each combination of conversion subprocess, parameter, value and modification type that you define as
            necessary to change there should be a list with four elements in 'params_cs_selection'. Each of the combination
            lists is composed of [['parameters'], 'cs', ['new_values'], 'mod_type']. It MUST be in this order, you can't
            change it by any means, since it will be identified by it's order later in the pipeline. The output structure you
            should always follow is [[combination], [combination], [combination], ...], and this list that comprehends
            all of the combinations of parameter, cs, value and modification type is the single element of 'params_cs_selection'.\n
            
            NEVER MAKE UP DATA, USE ONLY DATA FROM THE GIVEN LISTS. NEVER MODIFY THE SELECTED ENTRY, USE IT AS YOU
            FOUND IT IN THE LIST! FOR THE CONVERSION SUBPROCESSES YOU MUST USE THE ENTIRE CS NAME, IN THE FORMAT (CP@CIN@COUT),
            NEVER USE ONLY A PART OF IT. \n
            
            You must analyze the user's scenario request and decide all conversion subprocesses (cp@cin@cout) that should be modified,
            as well as the parameters necessary to be modified on each of the subprocesses. Each element of 'params_cs_selection'
            should contain a list with the parameters to be modified, the selected conversion subprocess to be modified 
            and a list with all the new values. 'parameters' and 'new_values' should have the same size. \n
            
            You have only two possibilities for the new values:
            1. You want to change something to 0, in this case you output a numerical 0 in the value;
            2. In any other case you should use percentual changes through decimal numbers, these decimals will be multiplied
            with the current value of the parameter.
            
            For the 'mod_type' you can use two different types, 'fixed' or 'multiplier'. 'fixed' tells the tool
            that you want the value to be exactly what you specified, 'multiplier' indicates that you want the
            current value to be multiplied by the value you defined. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            PARAMS: {params} \n
            CONVERSION_SUBPROCESSES: {CSs} \n
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["user_input","context","params","CSs"],
        )
    
    def execute(self) -> GraphState:
        if self.debug:
            print('---MODEL MODIFIER---')
            
        sheet_selector_prompt = self.get_sheet_selector_prompt_template()
        scenario_param_prompt = self.get_scenario_param_prompt_template()
        params_general_prompt = self.get_params_general_prompt_template()
        params_defined_prompt = self.get_params_defined_prompt_template()
        params_undefined_prompt = self.get_params_undefined_prompt_template()
        
        sheet_selector_chain = sheet_selector_prompt | self.json_model | JsonOutputParser()
        scenario_param_chain = scenario_param_prompt | self.json_model | JsonOutputParser()
        params_general_chain = params_general_prompt | self.json_model | JsonOutputParser()
        params_defined_chain = params_defined_prompt | self.json_model | JsonOutputParser()
        params_undefined_chain = params_undefined_prompt | self.json_model | JsonOutputParser()
    
        user_input = self.state['user_input']
        context = self.state['context']
        action_history = self.state['action_history']
        scen_modded = self.state['scen_modded']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        # TODO change to multipliers
        
        helper = HelperFunctions()

        params, CSs = helper.get_params_and_cs_list(self.base_model)
        scen_params = helper.get_scenario_params(self.base_model)
        
        llm_output = sheet_selector_chain.invoke({"user_input": user_input,
                                                  "scen_modded": scen_modded})
        sheet = llm_output['sheet']

        if llm_output['sheet'] == 'scenario':
            new_params = scenario_param_chain.invoke({"user_input": user_input,
                                                      "scen_params": scen_params})
        else:
            llm_output = params_general_chain.invoke({"user_input": user_input})

            if llm_output['parametrization_type'] == 'defined':
                new_params = params_defined_chain.invoke({"user_input": user_input,
                                                          "context": context,
                                                          "params": params,
                                                          "CSs": CSs})
            else:
                new_params = params_undefined_chain.invoke({"user_input": user_input,
                                                            "context": context,
                                                            "params": params,
                                                            "CSs": CSs})
        
        workbook = load_workbook(filename=self.base_model)

        if sheet == 'scenario':
            # The scenario sheet should be modified
            scen_sheet = workbook['Scenario']
            new_params = new_params['new_values']
            coords = ['','','','']
            
            for idx, row in enumerate(scen_sheet.rows):
                if idx == 0:
                    for i in range(len(row)):
                        if row[i].value == 'scenario_name':
                            coords[0] = row[i].coordinate[0]
                        if row[i].value == 'discount_rate':
                            coords[1] = row[i].coordinate[0]
                        if row[i].value == 'annual_co2_limit':
                            coords[2] = row[i].coordinate[0]
                        if row[i].value == 'co2_price':
                            coords[3] = row[i].coordinate[0]
                else:
                    if scen_sheet[f'{coords[0]}{idx+1}'].value == 'Base':
                        for i in range(len(coords)):
                            coords[i] = f'{coords[i]}{idx+1}'
                        break
            
            result = []
            col_names = ['discount_rate', 'annual_co2_limit', 'co2_price']
            for i in range(1,4):
                old_value = scen_sheet[coords[i]].value
                if old_value != new_params[i-1]:
                    scen_sheet[coords[i]].value = new_params[i-1]
                    result = result + [f'{col_names[i-1]} modified from {old_value} to {new_params[i-1]}']
                    
        else:
            # The conversion subprocess sheet should be modified
            cs_sheet = workbook['ConversionSubProcess']
            
            new_params = new_params['param_cs_selection']
            params_list = []
            
            for i in range(len(new_params)):
                parameter = new_params[i][0]
                if ' - ' in parameter:
                    parameter = parameter.split(' - ')[0]
                cs = new_params[i][1]
                new_value = new_params[i][2]
                mod_type = new_params[i][3]
                
                # Defined param format: [parameter, [cs_list], new_value]
                if type(cs) == list and len(cs) > 1:
                    data = []
                    for j in range(len(cs)):
                        elements = cs[j].split('@')
                        data.append([j+1,elements[0],elements[1],elements[2]])
                        table = tabulate(data, headers=["Index", "CP", "CIN", "COUT"])
                    
                    print('More than one match were found for one of the selected conversion subprocesses.')
                    print(table)
                    cs_select = int(input(f'Input the number of the correct CS (0 if none, and {len(cs)+1} if all):\n')) - 1
                    
                    if cs_select == len(cs):
                        for j in range(len(cs)):
                            params_list = params_list + [[parameter, cs[j], new_value, mod_type]]
                    elif cs_select >= 0:
                        params_list = params_list + [[parameter, cs[cs_select], new_value, mod_type]]
                elif type(cs) == list:
                    params_list = params_list + [[parameter, cs[0], new_value, mod_type]]
                    
                # Undefined param format: [[param_list], cs, [new_value_list]]
                if type(parameter) == list and len(parameter) > 1:
                    for i in range(len(parameter)):
                        params_list = params_list + [[parameter[i], cs, new_value[i], mod_type]]
                elif type(parameter) == list:
                    params_list = params_list + [[parameter[0], cs, new_value[0], mod_type]]
            
            #open workbook
            result = []
            for i in range(len(params_list)):
                parameter = params_list[i][0]
                if ' - ' in parameter:
                    parameter = parameter.split(' - ')[0]
                cs = params_list[i][1]
                split_cs = cs.split('@')
                new_value = params_list[i][2]
                mod_type = params_list[i][3]
                
                param_idx = '0'
                cs_idx = '0'
                
                full_cs_found = False
                cs_sub = ''
                for idx, row in enumerate(cs_sheet.rows):
                    if idx == 0:
                        for i in range(len(row)):
                            if row[i].value == parameter:
                                param_idx = row[i].coordinate
                    else:
                        if f'{row[0].value}@{row[1].value}@{row[2].value}' == cs:
                            cs_idx = row[0].coordinate
                            full_cs_found = True
                        elif f'{row[0].value}@{split_cs[1]}@{row[2].value}' == cs and not full_cs_found:
                            cs_idx = row[0].coordinate
                            cs_sub = f'{row[0].value}@{split_cs[1]}@{row[2].value}'
                if param_idx == '0' or cs_idx == '0':
                    result = result + [f'{parameter} of {cs} not found.']
                else:
                    if not full_cs_found and len(cs_sub) > 0:
                        cs = cs_sub
                    old_value = cs_sheet[f'{param_idx[0]}{cs_idx[1:]}'].value
                    if mod_type == 'multiplier':
                        if type(old_value) and '[' in old_value:
                            full_new_value = '['
                            split_value = old_value[1:-1].split(';')
                            for y in split_value:
                                year, value = y.split(' ')
                                full_new_value = f'{full_new_value}{year} {"%.2f" % (float(value)*new_value)};'
                            new_value = full_new_value[:-1] + ']'
                        else:
                            new_value = old_value * new_value
                    cs_sheet[f'{param_idx[0]}{cs_idx[1:]}'].value = new_value
                    
                    result = result + [f'{parameter} of {cs} modified from {old_value} to {new_value}']
                        
        try:
            workbook.save(filename=self.mod_model)
            if sheet == 'scenario':
                scen_modded = True
        except:
            result = ['Failed to save the modifications']
        
        if self.debug:
            print(f'FINAL RESULTS OF MODIFICATION:\n{result}\n')
        
        self.state['num_steps'] = num_steps
        self.state['scen_modded'] = scen_modded
        self.state['context'] = context + result
        self.state['actions_history'] = action_history + result
        
        return self.state
    
# TODO re-think the way the info flows from and to here

class ConsultModel(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at identifying from the user's USER_INPUT the correct consultation that
            should be made to the model to return the required information to the user. \n
            
            You'll receive CONVERSION_PROCESSES, CONVERSION_SUBPROCESSES, PARAMETERS and SCENARIO_INFO
            as context, as well as MODEL_INFOS, which is a list of information about the model that was
            already gathered. You should always focus on completing the necessary information, while
            avoiding to get information that already exists. \n
            
            CONVERSION_PROCESSES is a list with the name of all conversion processes. They are the general 
            processes of energy conversion and generation, better fragmentalized in CONVERSION_SUBPROCESSES. \n
            
            CONVERSION_SUBPROCESSES is presented to you in the format "'cp'@'cin'@'cout' - 'description'" and
            describes the processes of energy exchange, such as gas to heat, or gas to electricity, for each
            of the conversion processes. \n
            
            PARAMETERS is a list with all parameters that can be modeled for the conversion subprocesses along
            with their descriptions. \n
            
            SCENARIO_INFO is a dictionary with the relevant information about the scenario to consider
            if asked by the user. It contains the discount rate of the model, the anual limits of CO2 emission
            and the defined price for CO2. \n
            
            Beyond the already available information, you can ask for two other types of information,
            ['yearly_variation', 'cs_param_selection']. \n
            
            - yearly_variation: provides you with all parameters of specific conversion subprocesses that
            have yearly variation. The parameter is displayed as 'conversion_process'@'cin'@'cout'@'parameter', and
            the value comes as a list with years and values, such as [2016 10; 2020 20; 2030 30] for example. This
            would show you yearly variation of parameters for different conversion subprocesses;
            - cs_param_selection: allows you to choose a set of conversion subprocesses and parameters
            to get their values from the model. \n
            
            Your only output should be a JSON with three keys, 'consult_type', 'cs' and 'param'. 'consult_type' must
            be 'yearly_variation' or 'cs_param_selection, 'cs' and 'param' are optional, and should only be used
            in the 'cs_param_selection' case. \n
            
            Whenever you need to use 'cs_param_selection', you should get 'cs' from CONVERSION_SUBPROCESS and
            'param' from PARAMETERS. You MUST NOT modify the entries, you should use them exactly as they are given
            to you in the input lists, you are also NOT ALLOWED to guess CSs or params, use only the ones
            available in the lists. The values in 'cs' and 'param' should always be lists, even if you want
            to know a single value. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONVERSION_PROCESSES: {cp} \n
            CONVERSION_SUBPROCESSES: {cs} \n
            PARAMETERS: {param} \n
            SCENARIO_INFO: {scen_infos} \n
            MODEL_INFOS: {model_infos} \n
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["user_input","cp","cs","param","scen_infos","model_infos"],
        )
    
    def get_summarizing_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an specialist at sumarizing model information based on the user's USER_INPUT. \n
                
                You'll receive CONVERSION_PROCESSES, CONVERSION_SUBPROCESSES, PARAMETERS SCENARIO_INFO
                and CONSULTED_INFO as context. \n
                
                CONVERSION_PROCESSES is a list with the name of all conversion processes. They are the general 
                processes of energy conversion and generation, better fragmentalized in CONVERSION_SUBPROCESSES. \n
                
                CONVERSION_SUBPROCESSES is presented to you in the format "'cp'@'cin'@'cout' - 'description'" and
                describes the processes of energy exchange, such as gas to heat, or gas to electricity, for each
                of the conversion processes. \n
                
                PARAMETERS is a list with all parameters that can be modeled for the conversion subprocesses along
                with their descriptions. \n
                
                SCENARIO_INFO is a dictionary with the relevant information about the scenario to consider
                if asked by the user. It contains the discount rate of the model, the anual limits of CO2 emission
                and the defined price for CO2. \n
                
                CONSULTED_INFO contains possible extra information about the model that you may need to evaluate
                the answer. \n
                
                Your goal with all this context is to sumarize the information requested by the user about the
                model based on the USER_INPUT. Make the summarization the most natural possible, without citing the name
                of the source inputs. \n

                <|eot_id|><|start_header_id|>user<|end_header_id|>
                USER_INPUT: {user_input} \n
                CONVERSION_PROCESSES: {cp} \n
                CONVERSION_SUBPROCESSES: {cs} \n
                PARAMETERS: {param} \n
                SCENARIO_INFO: {scen_infos} \n
                CONSULTED_INFO: {info} \n
                Answer:
                <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """,
                input_variables=["user_input","cp","cs","param","scen_infos","info"],
        )
    
    def execute(self) -> GraphState:
        user_input = self.state['user_input']
        context = self.state['context']
        action_history = self.state['action_history']
        model_info = self.state['model_info']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        consult_prompt = self.get_prompt_template()
        sum_up_prompt = self.get_summarizing_prompt_template()
        
        consult_model_chain = consult_prompt | self.json_model | JsonOutputParser()
        sum_up_info_chain = sum_up_prompt | self.chat_model | StrOutputParser()
        
        helper = HelperFunctions()

        params, CSs = helper.get_params_and_cs_list(self.base_model)
        CPs = helper.get_conversion_processes(self.base_model)
        scen_infos = helper.get_scenario_params(self.base_model)
        
        llm_output = consult_model_chain.invoke({"user_input": user_input,
                                             "cp": CPs,
                                             "cs": CSs,
                                             "param": params,
                                             "scen_infos": scen_infos,
                                             "model_infos": model_info})
        
        if self.debug:
            print('---CONSULT MODEL---')
            print(f'ANALYZER OUTPUT: {llm_output}')

        info = helper.consult_info(llm_output, self.base_model)
        
        summed_up_info = sum_up_info_chain.invoke({"user_input": user_input,
                                                   "cp": CPs,
                                                   "cs": CSs,
                                                   "param": params,
                                                   "scen_infos": scen_infos,
                                                   "info": info})
        
        if self.debug:
            print(f'GATHERED INFO ABOUT THE MODEL: {summed_up_info}')
        
        self.state['num_steps'] = num_steps
        self.state['action_history'] = action_history + ['The model was consulted']
        self.state['model_info'] = model_info + [summed_up_info]
        self.state['context'] = context + [summed_up_info]
        
        return self.state

# TODO fill this agent

class CompareModel(AgentBase):
    # This should be able to compare any specific variable with the base result of the
    # model, list assets from the results, etc...
    pass

# TODO make this agent work

class PlotModel(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        action_history = self.state['action_history']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        if self.debug:
            print('---PLOTTER---')
            print(f'FINAL COMMAND: python cesm.py plot \n')
            
        self.state['num_steps'] = num_steps
        self.state['action_history'] = action_history + ['The requested data was successfully plotted!']
        
        return self.state

class ActionsAnalyzer(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a expert at checking the actions that were already taken regarding model
            manipulation, as well as checking if all requested information is already
            available for an output to be generated for the user. \n
            
            You have access to USER_INPUT, CONTEXT, MODEL_INFO and ACTION_HISTORY. USER_INPUT
            tells you what is the goal of the user, CONTEXT shows you the general information
            that may be needed to answer the user input, MODEL_INFO has the information about
            the model that may be needed to answer the user, and ACTION_HISTORY shows you what
            was already done to reach the goal of fulfilling the user's request. \n
            
            You must output a JSON with a single key 'ready_to_answer' that may be either true
            or false, depending on your judgement on the completeness of the data regarding
            the input. ALWAYS WRITE THE BOOLEAN IN LOWERCASE, OTHERWISE THE JSON PARSE WILL FAIL. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            MODEL_INFO: {model_info} \n
            ACTION_HISTORY: {action_history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","model_info","action_history"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        ## Get the state
        user_input = self.state['user_input']
        context = self.state['context']
        model_info = self.state['model_info']
        action_history = self.state['action_history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input,
                                       "context": context,
                                       "model_info": model_info,
                                       "action_history": action_history})
        
        if self.debug:
            print("---ACTIONS ANALYZER---")
            print(f'IS READY TO ANSWER: {llm_output["ready_to_answer"]}\n')
        
        self.state['next_query'] = llm_output
        self.state['num_steps'] = num_steps
        
        return self.state

# TODO the outputs should also indicate if the model was runned etc...
    
class OutputGenerator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at answering the user based on context given. \n
            
            Given the USER_INPUT and a CONTEXT, generate an answer for the query
            asked by the user. You should make use of the provided information
            to answer the user in the best possible way. If you think the answer
            does not answer the user completely, ask the user for the necessary
            information if possible. \n
            
            CHAT_HISTORY can also be used to gather context and information about
            past messages exchanged between you and the user. \n
            
            If and only if the context you use to answer the user provides you a
            data source you should display the provided sources as follows:
            Source:
            - <url/document>
            - <url/document>
            - And so on for how many sources are needed... \n
            
            NEVER output an empty source, if there is no provided source DON'T
            ADD THE SOURCE LIST. THE PRESENCE OF THE SOURCE LIST IS OPTIONAL,
            IF THERE ARE NO SOURCES IN THE DATA YOU USED TO GENERATE THE ANSWER, THEN
            DON'T CREATE THE SOURCE LIST. \n            
            
            It's important never to cite the variables you receive, answer the most
            naturally as possible trying to make it as it was a simple conversation. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            CHAT_HISTORY: {history}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.chat_model | StrOutputParser()
        
        ## Get the state
        user_input = self.state['user_input']
        context = self.state['context']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "context": context, "history": history})
        
        if self.debug:
            print("---GENERATE OUTPUT---")
            print(f'GENERATED OUTPUT:\n{llm_output}\n')
        
        if '\nSource:\n- None' in llm_output:
            llm_output = llm_output.replace('\nSource:\n- None','')
            
        self.state['scen_modded'] = False
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = llm_output
        
        return self.state
    
# Outdated

class ESToolSelector(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at reading the user QUERY and and the  routing it to the correct tool in our
            modelling system. \n

            Use the following criteria to decide how to route the query to one of our available tools: \n\n
            
            If the user asks for any modification on any particular model, select 'model_modifier'. \n
            
            If the user asks to plot anything, select 'data_plotter'. \n
            
            If the user asks to run a simulation of any particular model, select 'sim_runner'. \n

            You must output a JSON object with two keys:
            'selected_tool' containing one of the following values ['model_modificator', 'data_plotter', 'sim_runner'];
            'selected_model' containing the name of the model to be manipulated. \n
            
            If the user didn't provide a model name, fill the key 'selected_model' with 'NO_MODEL'. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY : {query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        query = self.state['user_input']
        num_steps = self.state['num_steps']
        num_steps += 1

        router = llm_chain.invoke({"query": query})
        router_decision = router['selected_tool']
        identified_model = router['selected_model']
        
        if self.debug:
            print("---ENERGY SYSTEM TOOL SELECTION---")
            print(f'QUERY: {query}')
            print(f'SELECTED TOOL: {router_decision}')
            print(f'IDENTIFIED MODEL: {identified_model}\n')
            
        self.state['selected_tool'] = router_decision
        self.state['identified_model'] = identified_model
        self.state['num_steps'] = num_steps
        
        return self.state

class ParamsIdentifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at identifying the correct conversion subprocess and the correct parameter
            selected by the user in his QUERY. \n
            
            As a context, you will receive two data arrays. PARAMS provides you the name of the parameters
            available to be selected. CONVERSION_SUBPROCESSES provides you the combination of 'cp' (conversion process name),
            'cin' (commodity in), 'cout' (commodity out) and 'scen' (scenario) in the format 'cp@cin@cout@scen'.\n
            
            Your goal is to output a JSON object containing three keys: 'param', 'value', 'cs_list'.
            'param' must receive the name of the selected parameter;
            'value' is the new value selected by the user;
            'cs_list' is a list with all matching conversion subprocesses (idealy only one if possible); \n
            
            NEVER MAKE UP DATA, USE ONLY DATA FROM THE GIVEN LIST. NEVER MODIFY THE SELECTED ENTRY, USE IT AS YOU
            FOUND IT IN THE LIST! \n
            
            If you can't find any match to the 'cp' name, leave the field 'cs_list' empty. If you can't find any match
            to the 'param' name, fill the field param with 'NOT_FOUND'. \n
            
            For the value required by the user, if the value is not directly stated in the QUERY, it will be
            available in the CONTEXT, use the data found there, never try to guess the desired value. If you can't find
            the value in the context leave the field 'value' empty. \n
            
            The field 'value' only accepts numeric input, unless the input given by the user contains [], in this
            case you should output it as a string. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            CONTEXT: {context} \n
            PARAMS: {params} \n
            CONVERSION_SUBPROCESSES: {CSs} \n
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["query","context","params","CSs"],
        )
    
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        if self.debug:
            print("---PARAM SELECTOR---")
            
        query = self.state['initial_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        self.state['num_steps'] = num_steps

        tmap = pd.ExcelFile('Models/DEModel.xlsx')
        df = pd.read_excel(tmap,"ConversionSubProcess")

        conversion_processes = np.asarray(df.iloc[:,0].dropna())
        mask = np.where(conversion_processes != 'DEBUG')
        conversion_processes = conversion_processes[mask]
        parameters = np.asarray(df.columns[4:])

        cs = np.asarray(df.iloc[:,0:4].dropna())
        mask = np.where(cs[:,0] != 'DEBUG')
        cs = cs[mask]
        conversion_subprocesses = np.empty((len(cs),1),dtype=object)

        for i in range(len(cs)):
            conversion_subprocesses[i] = f'{cs[i,0]}@{cs[i,1]}@{cs[i,2]}@{cs[i,3]}'

        output = llm_chain.invoke({"query": query, "context": context, "params": parameters, "CSs": conversion_subprocesses})
        
        if self.debug:
            print('---CONFIRM SELECTION---')
        # TODO make this work in the chat UI
        cs_list = output['cs_list']
        param = output['param']
        new_value = output['value']
        data = []
        for i in range(len(cs_list)):
            elements = cs_list[i].split('@')
            data.append([i+1,elements[0],elements[1],elements[2],elements[3]])
            table = tabulate(data, headers=["Index", "CP", "CIN", "COUT", "Scen"])
        
        if len(data) == 0:
            print('No matching conversion subprocess was found.')
            cs_confirm = 'N'
        elif len(data) == 1:
            print('The following matching conversion subprocess was found:\n')
            print(table)
            cs_confirm = input('Is that correct? (Y or N)\n')
            cs_select = 0 if cs_confirm == 'Y' else 'NONE'
        else:
            print('The following conversion subprocesses were found:\n')
            print(table)
            cs_select = int(input('Input the number of the correct CS (or 0 if it\'s none of these):\n')) - 1
            cs_confirm = 'Y' if cs_select != -1 else 'N'
        
        if cs_confirm == 'N':
            print('FINAL ANSWER: No matching selection.')
            self.state['cs'] = 'NO_MATCH'
            self.state['selection_is_valid'] = False
            self.state['parameter'] = 'NO_MATCH'
            return self.state
            
        if param in parameters:
            param_confirm = input(f'You want to modify the parameter {param}, is that correct? (Y or N)\n')
        else:
            print('No matching parameter was found.')
            param_confirm = 'N'
            
        if param_confirm == 'N':
            print('FINAL ANSWER: No matching selection.')
            self.state['cs'] = cs_list[cs_select]
            self.state['selection_is_valid'] = False
            self.state['parameter'] = 'NO_MATCH'
        else:
            print(f'FINAL ANSWER: CS: {cs_list[cs_select]}; Param: {param}')
            self.state['cs'] = cs_list[cs_select]
            self.state['new_value'] = new_value
            self.state['selection_is_valid'] = True
            self.state['parameter'] = param
        
        return self.state
    
class ScenarioIdentifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at identifying the correct scenario choosen by the user
            in his QUERY to have the simulation run. \n
            
            As a context, you will receive a data array called SCENARIOS, which contains
            all of the scenarios that are available to be simulated. \n
            
            Your goal is to output a JSON object containing one key called 'scenario_name' that contains
            the name of the scenario selected by the user. \n
            
            NEVER MAKE UP DATA, USE ONLY DATA FROM THE GIVEN LIST. If you can't find any match to the asked scenario,
            simply fill the key 'scenario_name' with 'NOT_FOUND'. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            SCENARIOS: {scenarios} \n
            Answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["query","scenarios"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        if self.debug:
            print('---SCENARIO SELECTOR---')
        
        query = self.state['initial_query']
        num_steps = self.state['num_steps']
        num_steps += 1
        self.state['num_steps'] = num_steps
        
        tmap = pd.ExcelFile('models/DEModel.xlsx')
        df = pd.read_excel(tmap,"Scenario")
        scenarios = np.asarray(df.iloc[:,0].dropna())
        
        output = llm_chain.invoke({'query': query, 'scenarios': scenarios})
        identified_scenario = output['scenario_name']
        print(f'IDENTIFIED SCENARIO: {identified_scenario}')
        #TODO make this work in the chat
        if identified_scenario == 'NOT_FOUND' or not(identified_scenario in scenarios):
            print('No valid scenario was identified in the request, here are the available scenarios:\n')
            for i in range(len(scenarios)):
                print(f'{i+1} - {scenarios[i]}')
            selection = int(input('Select the desired scenario to be run (select 0 if none of these):\n'))-1
            if selection != -1:
                identified_scenario = scenarios[selection]
            else:
                identified_scenario = 'NOT_FOUND'
        # if identified_scenario == 'NOT_FOUND' or not(identified_scenario in scenarios):
        #     kwargs = {
        #         'label': 'No valid scenario was identified in the request, here are the available scenarios:',
        #         'buttons': scenarios,
        #         'app': self.app,
        #         'callback': self.confirm_selection
        #     }
        #     input_getter = InputGetter(self.app, **kwargs)
        #     if not(self.selected_value in scenarios):
        #         identified_scenario = self.selected_value
        #     else:
        #         identified_scenario = 'NOT_FOUND'
        
        if identified_scenario == 'NOT_FOUND':
            message = 'No valid scenario was found'
            valid = False
        else:
            message = f'Selected scenario for simulation: {identified_scenario}'
            valid = True
            
        print(message)
        
        self.state['scenario'] = identified_scenario
        self.state['selection_is_valid'] = valid
        self.state['final_answer'] = message

        return self.state
    
class PlotIdentifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a specialist at identifying from the user's QUERY the correct plot type requested by
        the user and the desired variable the user wants to plot. \n
        
        As a context, you will receive two data arrays:
        PLOT_TYPES will provide you information about the available plot types;
        VARIABLES will provide you information about the available variables to be plotted. \n
        
        Your goal is to output a JSON OBJECT containing only two keys 'plot_type' and 'variable'.
        'plot_type' will receive the selected plot type from PLOT_TYPES, if you can't find the plot
        requested by the user in the list, fill the key with 'NOT_FOUND';
        'variable' will receive the selected variable from VARIABLES, if you can't find the
        variable requested by the user fill the key with 'NOT_FOUND'. \n
        
        NEVER MAKE UP DATA, USE ONLY DATA FROM THE GIVEN LISTS. \n

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        QUERY: {query} \n
        PLOT_TYPES: {plot_types} \n
        VARIABLES: {variables} \n
        Answer:
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["query","plot_types","variables"],
    )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        if self.debug:
            print('---PLOT SELECTOR---')
            
        num_steps = self.state['num_steps']
        num_steps += 1
        query = self.state['initial_query']
        
        plot_types = ['Bar', 'TimeSeries', 'Sankey', 'SingleValue']
        variables = ['TOTEX','OPEX','CAPEX','total_annual_co2_emission','cap_active','cap_new','cap_res','pin','pout']
        
        output = llm_chain.invoke({"query": query, "plot_types": plot_types, "variables": variables})
        identified_plot = output['plot_type']
        identified_variable = output['variable']
        print(f'IDENTIFIED PLOT: {identified_plot}\nIDENTIFIED VARIABLE: {identified_variable}')
        
        if identified_plot == 'NOT_FOUND' or not(identified_plot in plot_types):
            print('No valid plot type was identified in the request, here are the available plot types:\n')
            for i in range(len(plot_types)):
                print(f'{i+1} - {plot_types[i]}')
            selection = int(input('Select the desired type of plot (select 0 if none of these):\n'))-1
            if selection != -1:
                identified_plot = plot_types[selection]
            else:
                identified_plot = 'NOT_FOUND'
        # TODO
        if identified_variable == 'NOT_FOUND' or not(identified_variable in variables):
            print('No valid variable was identified in the request, here are the available variables:\n')
            for i in range(len(variables)):
                print(f'{i+1} - {variables[i]}')
            selection = int(input('Select the desired variable to be plotted (select 0 if none of these):\n'))-1
            if selection != -1:
                identified_variable = plot_types[variables]
            else:
                identified_variable = 'NOT_FOUND'
        
        if identified_plot == 'NOT_FOUND' or identified_variable == 'NOT_FOUND':
            message = 'No valid plot was identified'
            valid = False
        else:
            message = f'Selected plot: {identified_plot} for {identified_variable}'
            valid = True
        
        print(message)
        
        self.state['num_steps'] = num_steps
        self.state['plot_type'] = identified_plot
        self.state['variable'] = identified_variable
        self.state['selection_is_valid'] = valid
        self.state['final_answer'] = message
        
        return self.state