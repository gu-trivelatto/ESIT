from abc import ABC, abstractmethod
from llm_src.state import GraphState
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from datetime import datetime
from openpyxl import load_workbook
import pandas as pd
import numpy as np
from tabulate import tabulate
import os

class AgentBase(ABC):
    def __init__(self, chat_model, json_model, state: GraphState, debug):
        self.chat_model = chat_model
        self.json_model = json_model
        self.state = state
        self.debug = debug

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        pass

    def execute(self) -> GraphState:
        pass
    
class ResearchAgentBase(ABC):
    def __init__(self, chat_model, json_model, retriever, web_tool, state: GraphState, debug):
        self.retriever = retriever
        self.web_tool = web_tool
        self.chat_model = chat_model
        self.json_model = json_model
        self.state = state
        self.debug = debug
        
    def get_answer_analyzer_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at summarizing a bunch of data to extract only the important bits from it.

            Given the user's QUERY and the SEARCH_RESULTS, summarize as briefly as possible the information
            searched by the user. Don't give any preamble or introduction, go directly to the summary
            of the requested information.
            
            If it helps to provide a more precise answer, you can also make use of the CONTEXT.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            SEARCH_RESULTS: {search_results} \n
            CONTEXT: {context} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","search_results","context"],
        )

    @abstractmethod
    def get_first_prompt_template(self) -> PromptTemplate:
        pass
    
    def get_second_prompt_template(self) -> PromptTemplate:
        pass

    def execute(self) -> GraphState:
        pass

class TypeIdentifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at identifying the type of a query provided by the user among
            the types "general", "energy_system" and "mixed".

            "general": the query is related to some generic topic, it may consist of one or more
            points that require searching for information. \n
            
            "energy_system": the query is a direct command related to the energy system model, it can
            be a request to change parameters, plot data, run simulations, or anything on this lines.
            To be characterized as this class, it should need no external information. Names of
            simulations, scenarios, parameters and any other potential name is assumed to be know by our tools. \n
            
            "mixed": the query is related to the energy system model, but it requires external data for the
            command to be complete. It MUST be related to running anything related to the energy system,
            otherwise it is not mixed. \n
            
            You must output a JSON with a single key 'query_type' containing exclusivelly the 
            selected type. \n
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY : {query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
    
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        query = self.state['initial_query']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        gen = llm_chain.invoke({"query": query})
        selected_type = gen['query_type']
        
        if self.debug:
            print("---TYPE IDENTIFIER---")
            print(f'QUERY: {query}')
            print(f'IDENTIFIED_TYPE: {selected_type}\n')
        
        self.state['query_type'] = selected_type
        self.state['num_steps'] = num_steps
        
        return self.state

class ESToolSelector(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at reading the user QUERY and routing it to the correct tool in our
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
        
        query = self.state['initial_query']
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

class ModelSelector(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        # TODO make this work with the chat in a prettier way
        num_steps = self.state['num_steps']
        num_steps += 1
        
        print("No valid model was found for the requested action, the available models are:\n")
        
        available_models = next(os.walk('Models'), (None, None, []))[2]
        for i in range(len(available_models)):
            print(f'{i+1}: {available_models[i]}')
        
        selected_model = input('Please, inform the number of the desired model:\n')
        
        self.state['model'] = available_models[int(selected_model)-1]
        
        return self.state
    
class Mixed(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at reading the user QUERY and the available CONTEXT to decide if there
            is already enough information gathered to fulfill the energy system related command
            made by the user. \n
            
            You must be certain that you have all the data before deciding to send it to the
            modelling section of the pipeline. If any of the values asked by the user is not
            directly given by him, you can't consider the data complete unless you have the
            desired value in the CONTEXT. \n

            You must output a JSON object with a single key 'complete_data' containing a boolean
            on whether you have enough data for the user's request or not. \n
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY : {query} \n
            CONTEXT: {context} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","context"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        query = self.state['initial_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1

        decision = llm_chain.invoke({"query": query, "context": context})
        decision = decision['complete_data']
        
        if self.debug:
            print("---TOOL SELECTION---")
            print(f'QUERY: {query}')
            print(f'CONTEXT: {context}')
            print(f'DATA IS COMPLETE: {decision}\n')
            
        self.state['complete_data'] = decision
        self.state['num_steps'] = num_steps
        
        return self.state
    
class ToolSelector(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at reading a QUERY from a user and routing to our internal knowledge system\
            or directly to final answer. \n

            Use the following criteria to decide how to route the query to one of our available tools: \n\n
            
            If the user asks anything about LangSmith, you should use the 'RAG_retriever' tool.
            
            For any mathematical problem you should use 'calculator'. Be sure that you have all the necessary
            data before routing to this tool.

            If you are unsure or the person is asking a question you don't understand then choose 'web_search'

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
        
        query = self.state['next_query']
        num_steps = self.state['num_steps']
        num_steps += 1

        router = llm_chain.invoke({"query": query})
        router_decision = router['router_decision']
        
        print("---TOOL SELECTION---")
        print(f'QUERY: {query}')
        print(f'SELECTED TOOL: {router_decision}\n')
        
        self.state['selected_tool'] = router_decision
        self.state['num_steps'] = num_steps
        
        return self.state
    
class ResearchInfoRAG(ResearchAgentBase):
    def get_first_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a master at working out the best questions to ask our knowledge agent to get the best info for the customer.

            Given the INITIAL_QUERY, work out the best questions that will find the best \
            info for helping to write the final answer. Write the questions to our knowledge system not to the customer.

            Return a JSON with a single key 'questions' with no more than 3 strings of and no preamble or explaination.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            INITIAL_QUERY: {initial_query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["initial_query"],
        )
        
    def get_second_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n

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
            {"context": self.retriever , "question": RunnablePassthrough()}
            | rag_prompt
            | self.chat_model
            | StrOutputParser()
        )
        answer_analyzer_chain = answer_analyzer_prompt | self.chat_model | StrOutputParser()
        
        if self.debug:
            print("---RAG LANGSMITH RETRIEVER---")
            
        initial_query = self.state['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1

        questions = question_rag_chain.invoke({"initial_query": initial_query})
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
        
        processed_searches = answer_analyzer_chain.invoke({"query": initial_query, "search_results": rag_results, "context": context})
        
        self.state['context'] = context + [processed_searches]
        self.state['rag_questions'] = questions
        self.state['num_steps'] = num_steps
        
        return self.state
    
class ResearchInfoWeb(ResearchAgentBase):
    def get_first_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a master at working out the best keywords to search for in a web search to get the best info for the user.

            Given the INITIAL_QUERY, work out the best keywords that will find the info requested by the user
            The keywords should have between 3 and 5 words each, if the query allows for it.

            Return a JSON with a single key 'keywords' with no more than 3 keywords and no preamble or explaination.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            INITIAL_QUERY: {initial_query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["initial_query"],
        )
        
    def execute(self) -> GraphState:
        if self.debug:
            print("---RESEARCH INFO SEARCHING---")
            
        initial_query = self.state['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        prompt = self.get_first_prompt_template()
        answer_analyzer_prompt = self.get_answer_analyzer_prompt_template()
        
        llm_chain = prompt | self.json_model | JsonOutputParser()
        answer_analyzer_chain = answer_analyzer_prompt | self.chat_model | StrOutputParser()

        # Web search
        keywords = llm_chain.invoke({"initial_query": initial_query, "context": context})
        keywords = keywords['keywords']
        full_searches = []
        for idx, keyword in enumerate(keywords):
            temp_docs = self.web_tool.invoke({"query": keyword})
            if type(temp_docs) == list:
                web_results = "\n".join([d["content"] for d in temp_docs])
                web_results = Document(page_content=web_results)
            elif type(temp_docs) == dict:
                web_results = temp_docs["content"]
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

        processed_searches = answer_analyzer_chain.invoke({"query": initial_query, "search_results": full_searches, "context": context})
        
        if self.debug:
            print(f'FULL RESULTS: {full_searches}\n')
            print(f'PROCESSED RESULT: {processed_searches}')
        
        self.state['context'] = context + [processed_searches]
        self.state['num_steps'] = num_steps
        
        return self.state
    
class Calculator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at building JSON to do calculations using a calculator tool.
            
            You can only output a single format of JSON object consisting in two operands
            and the operation. The name of the only three keys are 'operation', 'op_1' and 'op_2' \n
            
            'operation' can only be [+,-,*,/,^]
            'op_1' and 'op_2' must be integers or float\n
            
            If you judge that the equation consists of more than one operation, solve only one,
            the calculator can be called multiple times and the other results will be solved
            later.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            INITIAL_QUERY: {initial_query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["initial_query"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
    
        query = self.state['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        parameters = llm_chain.invoke({"initial_query": query})
        operation = parameters['operation']
        op_1 = parameters['op_1']
        op_2 = parameters['op_2']

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
            str_result = 'Unable to execute the selected operation'
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
    
class ContextAnalyzer(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at deciding if the already available information is enough to
            fully answer the user query. \n
            
            Given a INITIAL_QUERY and the available CONTEXT, decide if the available information
            is already enough to answer the query proposed by the user. \n
            
            Your job is to coordinate the usage of many tools, one at a time. To do this you will
            decide what information you need next, with the restriction that you can only get one
            information per iteration, and request it to the pipeline. \n
            
            Your output should be a JSON object containing three keys, 'ready_to_answer',
            'next_query' and 'user_input'. 'ready_to_answer' is a boolean that indicates if all
            necessary info is present, 'next_query' is a query that you should develop so the next
            agent in the pipeline can search for the required information and 'user_input' is also
            a boolean that indicates if the question can ONLY be answered by the user. If there is
            any chance of finding the information without asking further questions to the user, leave
            this field as false.\n
            
            In the following situations you must output 'next_query' as "<KEEP_QUERY>":
            - User asks to modify parameters or characteristics of an energy system model;
            - Plotting, they don't require extra information, the tools can handle it perfectly;
            - User asks you to run a new simulation on an energy modeling system;
            - User gives you a direct command related to modelling;
            - The user asks anything about LangSmith (understand that as having the word LangSmith) \n
            
            You also have access to the last NEXT_QUERY you generated, to avoid repeating yourself.
            Never output the same 'next_query' that you've already asked in NEXT_QUERY. \n
            
            Consider that for you boolean answer the words false and true should always be written
            in full lower case. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            INITIAL_QUERY: {initial_query} \n
            CONTEXT: {context} \n
            NEXT_QUERY: {next_query} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["initial_query","context","next_query"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        if self.debug:
            print("---CONTEXT ANALYZER---")
            
        ## Get the state
        initial_query = self.state['initial_query']
        next_query = self.state['next_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1

        output = llm_chain.invoke({"initial_query": initial_query,
                                   "next_query": next_query,
                                   "context": context
                                   })
        
        if output['next_query'] == '<KEEP_QUERY>':
            output['next_query'] = self.state['initial_query']
        
        if output['user_input']:
            context = context + [output['next_query']]
        
        self.state['next_query'] = output
        self.state['context'] = context
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
        
        tmap = pd.ExcelFile('Models/DEModel.xlsx')
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
    
class ModelModifier(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        if self.debug:
            print('---MODEL MODIFIER---')
    
        model = self.state['model']
        parameter = self.state['parameter']
        cs = self.state['cs']
        new_value = self.state['new_value']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        model_file = model if '.xlsx' in model else f'{model}.xlsx'
        workbook = load_workbook(filename=f'Models/{model_file}')
        cs_sheet = workbook['ConversionSubProcess']
        
        #open workbook
        param_idx = '0'
        cs_idx = '0'
        for idx, row in enumerate(cs_sheet.rows):
            if idx == 0:
                for i in range(len(row)):
                    if row[i].value == parameter:
                        param_idx = row[i].coordinate
            else:
                if f'{row[0].value}@{row[1].value}@{row[2].value}@{row[3].value}' == cs:
                    cs_idx = row[0].coordinate
        if param_idx == '0' or cs_idx == '0':
            final_answer = 'Selected param or cs not found.'
            if self.debug:
                print('Selected param or cs not found.')
        else:
            old_value = cs_sheet[f'{param_idx[0]}{cs_idx[1:]}'].value
            cs_sheet[f'{param_idx[0]}{cs_idx[1:]}'].value = new_value
            workbook.save(filename="Models/DEModel_modified.xlsx")
            final_answer = f'Value successfully modified from {old_value} to {new_value}'
            if self.debug:
                print(f'Cell: {param_idx[0]}{cs_idx[1:]}')
                print(final_answer)
        
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = final_answer
        
        return self.state
    
class SimRunner(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        num_steps = self.state['num_steps']
        num_steps += 1
        model = self.state['model']
        scenario = self.state['scenario']
        
        if self.debug:
            print('---SIMULATION RUNNER---')
            print(f'FINAL COMMAND: python cesm.py run {model} {scenario}\n')
            
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = 'The requested simulation was successfully submited!'
        
        return self.state
    
class Plotter(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        num_steps = self.state['num_steps']
        num_steps += 1
        model = self.state['model']
        scenario = self.state['scenario']
        plot_type = self.state['plot_type']
        variable = self.state['variable']
        
        if self.debug:
            print('---PLOTTER---')
            print(f'FINAL COMMAND: python cesm.py plot {model} {scenario} {plot_type} {variable} \n')
            
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = 'The requested data was successfully plotted!'
        
        return self.state
    
class DateGetter(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    
    def execute(self) -> GraphState:
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        current_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        result = f'The current date and time are {current_date}'
        
        if self.debug:
            print("---DATE GETTER TOOL---")
            print(f'CURRENT DATE: {current_date}\n')

        self.state['context'] = context + [result]
        self.state['num_steps'] = num_steps

        return self.state
    
class OutputGenerator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a specialist at answering the user based on context given. \n
            
            Given the INITIAL_QUERY and a CONTEXT, generate an answer for the query
            asked by the user. You should make use of the provided information
            to answer the user in the best possible way. If you think the answer
            does not answer the user completely, ask the user for the necessary
            information if possible. \n
            
            It's important never to cite that you got it from a context, the user should
            think that you know the information.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            INITIAL_QUERY: {initial_query} \n
            CONTEXT: {context} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["initial_query","context"],
        )
        
    def execute(self) -> GraphState:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.chat_model | StrOutputParser()
        
        ## Get the state
        initial_query = self.state['initial_query']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1

        answer = llm_chain.invoke({"initial_query": initial_query,
                                                "context": context})
        if self.debug:
            print("---GENERATE OUTPUT---")
            print(f'GENERATED OUTPUT:\n{answer}\n')
            
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = answer
        
        return self.state
    
class EmptyNode(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return super().get_prompt_template()
    def execute(self) -> GraphState:
        return super().execute()