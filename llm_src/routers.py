from os import walk
from abc import ABC, abstractmethod
from llm_src.state import GraphState

class RouterBase(ABC):
    def __init__(self, state: GraphState, debug):
        self.state = state
        self.debug = debug
    
    @abstractmethod
    def execute(self) -> str:
        pass

class SelectionValidator(RouterBase):
    def execute(self) -> str:
        selection_is_valid = self.state['selection_is_valid']
        selected_tool = self.state['selected_tool']
        
        if selection_is_valid:
            selection = selected_tool
        else:
            selection = "end_not_valid"
        
        return selection
        
class RouteToType(RouterBase):
    def execute(self) -> str:
        """
        Route to the right path based on query type.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        type = self.state['query_type']
        
        if type == 'general':
            message = "---ROUTE QUERY TO GENERAL PATH---"
            selection = "general"
        elif type == 'energy_system':
            message = "---ROUTE QUERY TO ENERGY SYSTEM PATH---"
            selection = "energy_system"
        elif type == 'mixed':
            message = "---ROUTE QUERY TO MIXED PATH---"
            selection = "mixed"

        if self.debug:
            print(message)
            
        return selection

class RouteToType(RouterBase):
    def execute(self) -> str:
        """
        Route to the right path based on query type.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        type = self.state['query_type']
        
        if type == 'general':
            message = "---ROUTE QUERY TO GENERAL PATH---"
            selection = "general"
        elif type == 'energy_system':
            message = "---ROUTE QUERY TO ENERGY SYSTEM PATH---"
            selection = "energy_system"
        elif type == 'mixed':
            message = "---ROUTE QUERY TO MIXED PATH---"
            selection = "mixed"
        
        if self.debug:
            print(message)
            
        return selection

class RouteFromMix(RouterBase):
    def execute(self) -> str:
        data_completeness = self.state['complete_data']

        if data_completeness:
            message = "---APPLY COMMAND---"
            selection = "complete_data"
        else:
            message = "---GATHER MORE CONTEXT---"
            selection = "needs_data"
            
        if self.debug:
            print("---ROUTE TO MIX---")
            print(message)
            
        return selection

class ValidateSelectedModel(RouterBase):
    def execute(self) -> str:
        identified_model = self.state['identified_model']
        available_models = next(walk('Models'), (None, None, []))[2]
        
        if identified_model == 'NO_MODEL' or not(f'{identified_model}.xlsx' in available_models):
            selection = 'select_model'
        else:
            selection = 'model_is_valid'
            
        return selection

class RouteToESTool(RouterBase):
    def execute(self) -> str:
        """
        Route to the necessary tool.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        selection = self.state['selected_tool']
        
        if selection == 'data_plotter':
            message = "---ROUTE QUERY TO DATA PLOTTER---"
            selection = "data_plotter"
        elif selection == 'sim_runner':
            message = "---ROUTE QUERY TO SIMULATION RUNNER---"
            selection = "sim_runner"
        elif selection == 'model_modifier':
            message = "---ROUTE QUERY TO MODEL MODIFIER---"
            selection = "model_modifier"
            
        if self.debug:
            print(message)
            
        return selection
    
class RouteToTool(RouterBase):
    def execute(self) -> str:
        """
        Route to the necessary tool.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        selection = self.state['selected_tool']
        
        if selection == 'RAG_retriever':
            message = "---ROUTE QUERY TO RAG RETRIEVER---"
            selection = "RAG_retriever"
        elif selection == 'web_search':
            message = "---ROUTE QUERY TO WEB SEARCH---"
            selection = "web_search"
        elif selection == 'calculator':
            message = "---ROUTE QUERY TO CALCULATOR---"
            selection = "calculator"
            
        if self.debug:
            print(message)
            
        return selection

class RouteToIterate(RouterBase):
    def execute(self) -> str:
        next_query = self.state['next_query']

        if next_query['ready_to_answer'] or next_query['user_input']:
            message = "---GENERATE FINAL ANSWER---"
            selection = "ready_to_answer"
        else:
            message = "---GATHER MORE CONTEXT---"
            selection = "need_context"
            
        if self.debug:
            print("---ROUTE TO ITERATE---")
            print(message)
            
        return selection