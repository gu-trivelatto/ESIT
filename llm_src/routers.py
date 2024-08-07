from abc import ABC, abstractmethod
from llm_src.state import GraphState

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
        type = self.state['input_type']
        
        message = "---TYPE ROUTER---\nROUTE TO: "
        
        if type == 'general':
            message += "General branch\n"
            selection = "general"
        elif type == 'energy_system':
            message += "Energy System branch\n"
            selection = "energy_system"
        elif type == 'mixed':
            message += "Mixed branch\n"
            selection = "mixed"

        if self.debug:
            print(message)
            
        return selection

class MixedRouter(BaseRouter):
    def execute(self) -> str:
        data_completeness = self.state['complete_data']

        message = "---MIXED ROUTER---\nROUTE TO: "

        if data_completeness:
            message += "Run command\n"
            selection = "complete_data"
        else:
            message += "Gather more context\n"
            selection = "needs_data"
            
        if self.debug:
            print(message)
            
        return selection

class ESActionRouter(BaseRouter):
    def execute(self) -> str:
        """
        Route to the necessary action.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        selection = self.state['next_action']
        
        message = "---ENERGY SYSTEM ACTION ROUTER---\nROUTE TO: "
        
        if selection == 'run':
            message += "Run model\n"
            selection = "run"
        elif selection == 'modify':
            message += "Modify model\n"
            selection = "modify"
        elif selection == 'consult':
            message += "Consult model\n"
            selection = "consult"
        elif selection == 'compare':
            message += "Compare results\n"
            selection = "compare"
        elif selection == 'plot':
            message += "Plot results\n"
            selection = "plot"
        elif selection == 'no_action':
            message += "Generate output\n"
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
        
        message = "---TOOL ROUTER---\nROUTE TO: "
        
        if selection == 'RAG_retriever':
            message += "RAG Retriever\n"
            selection = "RAG_retriever"
        elif selection == 'web_search':
            message += "Web Search\n"
            selection = "web_search"
        elif selection == 'calculator':
            message += "Calculator\n"
            selection = "calculator"
        elif selection == "user_input":
            message += "User input\n"
            selection = "user_input"
            
        if self.debug:
            print(message)
            
        return selection

# TODO this router should be used also for the Actions router

class ContextRouter(BaseRouter):
    def execute(self) -> str:
        query = self.state['next_query']

        message = "---CONTEXT ROUTER---\nROUTE TO: "

        if query['ready_to_answer']:
            message += "Final answer generation\n"
            selection = "ready_to_answer"
        else:
            message += "Gather more context\n"
            selection = "need_context"
            
        if self.debug:
            print(message)
            
        return selection