from abc import ABC, abstractmethod
from llm_src.state import GraphState
from llm_src.helper import HelperFunctions

class BaseRouter(ABC):
    def __init__(self, state: GraphState, debug):
        self.state = state
        self.debug = debug
        self.helper = HelperFunctions()
    
    @abstractmethod
    def execute(self) -> str:
        pass
        
class BypassRouter(BaseRouter):
    def execute(self) -> str:
        """
        Route either to the tool or to the output.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        is_conversation = self.state['is_conversation']
        
        message = "---BYPASS ROUTER---\nROUTE TO: "
        
        if is_conversation:
            message += "Skip to output\n"
            selection = "output"
        else:
            message += "Use tool\n"
            selection = "tool"

        if self.debug:
            self.helper.save_debug(message)
            
        return selection

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
            self.helper.save_debug(message)
            
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
            self.helper.save_debug(message)
            
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
            self.helper.save_debug(message)
            
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
        selection = self.state['next_query']['tool']
        
        message = "---TOOL ROUTER---\nROUTE TO: "
        
        if selection == 'web_search':
            message += "Web Search\n"
            selection = "web_search"
        elif selection == 'calculator':
            message += "Calculator\n"
            selection = "calculator"
        elif selection == "user_input":
            message += "User input\n"
            selection = "direct_output"
        elif selection == "no_action":
            message += "No actions\n"
            selection = "direct_output"
            
        if self.debug:
            self.helper.save_debug(message)
            
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
            self.helper.save_debug(message)
            
        return selection
    
class InfoTypeRouter(BaseRouter):
    def execute(self) -> str:
        type = self.state['retrieval_type']

        message = "---INFO TYPE ROUTER---\nROUTE TO: "

        if type == 'paper':
            message += "Paper retrieval\n"
        elif type == 'model':
            message += "Model retrieval\n"
            
        if self.debug:
            self.helper.save_debug(message)
            
        return type

class TranslationRouter(BaseRouter):
    def execute(self) -> str:
        target_language = self.state['target_language']
        
        message = "---TRANSLATOR ROUTER---\nROUTE TO: "
        
        if target_language.lower() == 'english':
            message += "print output"
            translate = False
        else:
            message += "translate output\n"
            translate = True
        
        return translate