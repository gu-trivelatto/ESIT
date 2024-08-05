import os
import pickle
import customtkinter

from abc import ABC
from tkinter import *
from langgraph.graph import StateGraph
from llm_src.chat_llm import GraphBuilder

# pip install customtkinter

class Chat(ABC):
    def __init__(self, graph: StateGraph, recursion_limit):
        try:
            self.graph = graph
            os.remove("chat_history.pkl")
        except:
            self.graph = graph
        self.recursion_limit = recursion_limit

    def invoke(self, input) -> str:
        # run the agent
        try:
            with open("chat_history.pkl", "rb") as f:
                history = pickle.load(f)
        except:
            history = []
        history.append({"role": "user", "content": input})

        inputs = {"initial_query": history,
                  "next_query": [],
                  "num_steps": 0,
                  "context": [],
                  "model_info": [],
                  "actions_history": [],
                  "history": history,
                  "scen_ready": False}
        for output in self.graph.stream(inputs, {"recursion_limit": self.recursion_limit}):
            for key, value in output.items():
                print(f"Finished running <{key}> \n")
        return value['final_answer']
                
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        customtkinter.set_appearance_mode('dark')
        
        self.title("ESMChat")
        self.geometry("1200x900")
        
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        self.textbox = customtkinter.CTkTextbox(master=self,
                                                corner_radius=8,
                                                wrap='word',
                                                state='disabled')
        self.textbox.grid(row=0, column=1, rowspan=3, columnspan=3, padx=20, pady=(20, 0), sticky="nsew")

        self.entry = customtkinter.CTkTextbox(master=self, height=80)
        self.entry.grid(row=3, column=1, columnspan=2, padx=20, sticky="ew")
        self.entry.bind('<Return>', self.on_enter)
        self.entry.bind('<Shift-Return>', self.new_line)
        self.entry.focus_set()
        self.button = customtkinter.CTkButton(master=self, command=self.button_callback, text="Insert Text", height=80)
        self.button.grid(row=3, column=3, padx=20, pady=20, sticky="ew")
        
        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
    
    def submit_message(self):
        self.input_text = self.entry.get('0.0', END)
        new_text = "\n\nUSER:\n" + self.input_text
        self.textbox.configure(state="normal")
        self.textbox.insert(END, new_text)
        self.textbox.yview_moveto(1)
        self.textbox.configure(state="disabled")
        self.entry.delete("0.0", END)
        self.after(50, self.call_llm)
    
    def button_callback(self):
        self.submit_message()
    
    def on_enter(self, event):
        self.submit_message()
        return "break"
    
    def new_line(self, event):
        self.entry.insert(END, "\n")
        self.entry.yview_moveto(1)
        return "break"
    
    def call_llm(self):
        answer = self.chat.invoke(self.input_text)
        new_text = "\nASSISTANT:\n" + answer
        self.textbox.configure(state="normal")
        self.textbox.insert("end", new_text)
        self.textbox.yview_moveto(1)
        self.textbox.configure(state="disabled")
        
    def set_chat(self, chat):
        self.chat = chat
        
if __name__ == '__main__':
    app = App()
    graph = GraphBuilder('models', app, True).build()
    chat = Chat(graph, 25)
    app.set_chat(chat)
    app.mainloop()
    