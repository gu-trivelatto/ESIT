import pickle

from abc import ABC, abstractmethod
from llm_src.state import GraphState

class PrinterBase(ABC):
    def __init__(self, state: GraphState, debug):
        self.state = state
        self.debug = debug
    
    @abstractmethod
    def execute(self) -> None:
        pass

class StatePrinter(PrinterBase):
    def execute(self) -> None:
        """print the state"""
        if self.debug:
            print("------------------STATE PRINTER------------------")
            print(f"Num Steps: {self.state['num_steps']} \n")
            print(f"Initial Query: {self.state['user_input']} \n" )
            print(f"Context: {self.state['context']} \n" )
        return

class FinalAnswerPrinter(PrinterBase):
    def execute(self) -> None:
        """prints final answer"""
        if self.debug:
            print("------------------FINAL ANSWER------------------")
            print(f"Final Answer: {self.state['final_answer']} \n")
            
        history = self.state['history']
        history.append({"role": "assistant", "content": self.state['final_answer']})
        
        with open("chat_history.pkl", "wb") as f:
            pickle.dump(history, f)
        
        return self.state