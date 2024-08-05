from abc import ABC, abstractmethod
from llm_src.state import GraphState

# TODO standardize router names
# TODO standardize prints and variables naming

class BaseRouter(ABC):
    def __init__(self, state: GraphState, debug):
        self.state = state
        self.debug = debug
    
    @abstractmethod
    def execute(self) -> str:
        pass
        
class TypeRouter(BaseRouter):
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

class MixedRouter(BaseRouter):
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

class ESToolRouter(BaseRouter):
    def execute(self) -> str:
        """
        Route to the necessary tool.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        selection = self.state['next_action']
        
        if selection == 'run':
            message = "---ROUTE QUERY TO SIM RUNNER---"
            selection = "run"
        elif selection == 'modify':
            message = "---ROUTE QUERY TO MODEL MODIFIER---"
            selection = "modify"
        elif selection == 'consult':
            message = "---ROUTE QUERY TO MODEL CONSULT---"
            selection = "consult"
        elif selection == 'compare':
            message = "---ROUTE QUERY TO MODEL COMPARE---"
            selection = "compare"
        elif selection == 'plot':
            message = "---ROUTE QUERY TO PLOT MODEL---"
            selection = "plot"
        elif selection == 'no_action':
            message = "---ROUTE QUERY TO OUTPUT---"
            selection = "no_action"
            
        if self.debug:
            print(message)
            
        return selection
    
class ToolRouter(BaseRouter):
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
        elif selection == "user_input":
            message = "---ROUTE TO USER INPUT---"
            selection = "user_input"
            
        if self.debug:
            print(message)
            
        return selection

# TODO this router should be used also for the Actions router

class ContextRouter(BaseRouter):
    def execute(self) -> str:
        next_query = self.state['next_query']
        query = next_query[-1]
        
        print(query)

        if query['ready_to_answer']:
            message = "---GENERATE FINAL ANSWER---"
            selection = "ready_to_answer"
        else:
            message = "---GATHER MORE CONTEXT---"
            selection = "need_context"
            
        if self.debug:
            print("---ROUTE TO ITERATE---")
            print(query['next_query'])
            print(message)
            
        return selection