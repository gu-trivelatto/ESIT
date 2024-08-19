import pickle

from abc import ABC, abstractmethod
from llm_src.state import GraphState
from llm_src.helper import HelperFunctions

class PrinterBase(ABC):
    def __init__(self, state: GraphState, debug):
        self.state = state
        self.debug = debug
        self.helper = HelperFunctions()
    
    @abstractmethod
    def execute(self) -> None:
        pass

class StatePrinter(PrinterBase):
    def execute(self) -> None:
        """print the state"""
        if self.debug:
            self.helper.save_debug("------------------STATE PRINTER------------------")
            self.helper.save_debug(f"Num Steps: {self.state['num_steps']} \n")
            self.helper.save_debug(f"Initial Query: {self.state['user_input']} \n" )
            self.helper.save_debug(f"Consolidated Query: {self.state['consolidated_input']} \n")
            self.helper.save_debug(f"Context: {self.state['context']} \n" )
            self.helper.save_debug(f"Past actions: {self.state['action_history']} \n")
        return

class FinalAnswerPrinter(PrinterBase):
    def execute(self) -> None:
        """prints final answer"""
        if self.debug:
            self.helper.save_debug("------------------FINAL ANSWER------------------")
            self.helper.save_debug(f"Final Answer: {self.state['final_answer']} \n")
            
        history = self.state['history']
        history.append({"role": "assistant", "content": self.state['final_answer']})
        
        with open("chat_history.pkl", "wb") as f:
            pickle.dump(history, f)
        
        return self.state