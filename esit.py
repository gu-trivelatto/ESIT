import os
import time
import click
import pickle
import customtkinter

from abc import ABC
from tkinter import *
from threading import Thread
from llm_src.state import GraphState
from langgraph.graph import StateGraph
from llm_src.chat_llm import GraphBuilder
from llm_src.helper import HelperFunctions

class Chat(ABC):
    def __init__(self, graph: StateGraph, recursion_limit, debug):
        try:
            os.remove("metadata/chat_history.pkl")
        except:
            pass
        self.graph = graph
        self.recursion_limit = recursion_limit
        self.debug = debug
        self.helper = HelperFunctions()
        self.helper.save_simulation_status('not_runned')

    def invoke(self, input) -> str:
        # run the agent
        try:
            with open("metadata/chat_history.pkl", "rb") as f:
                history = pickle.load(f)
        except:
            history = []
        history.append({"role": "user", "content": input})
        self.helper.save_history(history)

        inputs = GraphState.initialize(input, history)
        for output in self.graph.stream(inputs, {"recursion_limit": self.recursion_limit}):
            with open('metadata/chat_control.log', 'r') as f:
                control_flag = f.read()
            if control_flag == 'aborting':
                value['final_answer'] = 'Generation aborted'
                break
            for key, value in output.items():
                if self.debug:
                    self.helper.save_debug(f"Finished running <{key}> \n")
        return value['final_answer']
                
class App(customtkinter.CTk):
    def __init__(self, debug, font_size):
        super().__init__()
        
        customtkinter.set_appearance_mode('dark')
        self.debug = debug
        self.font_size = font_size
        self.helper = HelperFunctions()
        
        self.title("ESIT")
        if self.debug:
            self.geometry("1800x900")
        else:
            self.geometry("900x900")
        
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        self.textbox = customtkinter.CTkTextbox(master=self,
                                                corner_radius=8,
                                                wrap='word',
                                                state='disabled',
                                                font=("", self.font_size))
        self.textbox.grid(row=0, column=1, rowspan=3, columnspan=3, padx=20, pady=(20, 0), sticky="nsew")

        self.entry = customtkinter.CTkTextbox(master=self, height=80, font=("", self.font_size), wrap='word')
        self.entry.grid(row=3, column=1, columnspan=2, padx=20, sticky="ew")
        self.entry.bind('<Return>', self.on_enter)
        self.entry.bind('<Shift-Return>', self.new_line)
        self.entry.focus_set()
        self.button = customtkinter.CTkButton(master=self,
                                              command=self.button_callback,
                                              text="Send",
                                              height=80,
                                              font=("", self.font_size))
        self.button.grid(row=3, column=3, padx=20, pady=20, sticky="ew")
        
        if self.debug:
            # create sidebar frame with widgets
            self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=0)
            self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
            self.sidebar_frame.grid_rowconfigure(0, weight=1)
            self.sidebar_frame.grid_columnconfigure(0, weight=1)
            self.debug_textbox = customtkinter.CTkTextbox(master=self.sidebar_frame,
                                                          corner_radius=8,
                                                          wrap='word',
                                                          state='disabled',
                                                          width=900,
                                                          font=("Ubuntu Mono", self.font_size))
            self.debug_textbox.grid(row=0, column=0, rowspan=4, columnspan=4, padx=20, pady=20, sticky="nsew")
            
            # Tracking whether the user is at the bottom for each textbox
            self.is_user_at_bottom = True

            # Bind scroll events for each textbox
            self.debug_textbox.bind("<MouseWheel>", self.on_user_scroll)
            self.debug_textbox.bind("<KeyPress>", self.on_user_scroll)
            self.debug_textbox.bind("<Button-4>", self.on_user_scroll)  # Linux scroll up
            self.debug_textbox.bind("<Button-5>", self.on_user_scroll)  # Linux scroll down
        
        self.available = True
        self.chat_running = False
        self.is_first_message = True
    
    def on_user_scroll(self, event):
        # Detect when user scrolls manually and stop automatic scrolling
        self.is_user_at_bottom = False
        if self.debug_textbox.yview()[1] == 1.0:
            self.is_user_at_bottom = True
    
    def update_debug(self):
        try:
            os.remove('metadata/debug.log')
        except:
            pass
        self.debug_textbox.configure(state="normal")
        self.debug_textbox.delete('0.0', END)
        self.debug_textbox.configure(state="disabled")
        last_position = 0
        
        while self.chat_running:
            try:
                with open('metadata/debug.log') as f:
                    f.seek(last_position)
                    new_log = f.read()
                    last_position = f.tell()
                
                if new_log:
                    self.debug_textbox.configure(state="normal")
                    self.debug_textbox.insert(END, new_log)
                    self.debug_textbox.configure(state="disabled")
                    if self.is_user_at_bottom:
                        self.debug_textbox.yview_moveto(1.0)  # Scroll to the bottom
            except:
                pass
            
    def loading_animation(self):
        states = ['   ', '.  ', '.. ', '...']
        self.animate_index = 0  # Track the current index in the states list

        def update_text():
            if self.chat_running:
                try:
                    with open('metadata/status.log', 'r') as f:
                        status = f.read()
                except:
                    status = 'Processing'
                try:
                    with open('metadata/chat_control.log', 'r') as f:
                        control_flag = f.read()
                    if control_flag == 'aborting':
                        status = 'Aborting'
                except:
                    pass
                self.entry.configure(state="normal")
                self.entry.delete('0.0', END)
                self.entry.insert('0.0', f'{status}{states[self.animate_index]}')
                self.entry.configure(state="disabled")
                self.animate_index = (self.animate_index + 1) % len(states)
                self.after(500, update_text)  # Schedule the next update
            else:
                self.entry.configure(state="normal")
                self.entry.delete('0.0', END)
                self.button.configure(text='Send')

        update_text()
    
    def submit_message(self):
        self.button.configure(text='Abort')
        self.input_text = self.entry.get('0.0', END)
        if self.is_first_message:
            new_text = "USER:\n" + self.input_text
        else:
            new_text = "\n\nUSER:\n" + self.input_text
        self.textbox.configure(state="normal")
        self.textbox.insert(END, new_text)
        self.textbox.yview_moveto(1)
        self.textbox.configure(state="disabled")
        self.entry.delete("0.0", END)
        Thread(target=self.call_llm).start()
        self.chat_running = True
        Thread(target=self.loading_animation).start()
        if self.debug:
            Thread(target=self.update_debug).start()
        self.is_first_message = False
    
    def button_callback(self):
        if self.button._text == 'Send':
            self.submit_message()
            with open('metadata/chat_control.log', 'w') as f:
                f.write('running')
        elif self.button._text == 'Abort':
            with open('metadata/chat_control.log', 'w') as f:
                f.write('aborting')
    
    def on_enter(self, event):
        if self.available:
            self.submit_message()
        return "break"
    
    def new_line(self, event):
        self.entry.insert(END, "\n")
        self.entry.yview_moveto(1)
        return "break"
        
    def call_llm(self):
        st = time.time()
        self.entry.configure(state="disabled")
        self.available = False
        try:
            answer = self.chat.invoke(self.input_text)
        except Exception as e:
            self.helper.save_debug(e)
            answer = 'An error has ocurred, please try again'
        new_text = "\nASSISTANT:\n" + answer
        self.textbox.configure(state="normal")
        self.textbox.insert("end", new_text)
        self.textbox.yview_moveto(1)
        self.textbox.configure(state="disabled")
        self.entry.configure(state="normal")
        self.available = True
        with open('metadata/chat_control.log', 'w') as f:
            f.write('idle')
        self.chat_running = False
        print(f"Execution finished in {time.time()-st:.2f} seconds")
        
        
        
    def set_chat(self, chat):
        self.chat = chat

@click.command()
@click.option('-d', '--debug', is_flag=True, help='Activate the debugger.')
def main(debug):
    print("Welcome to the Energy System Insight Tool (ESIT)")
    # TODO modify the way we get the path in CESM/core/input_parser.py
    app = App(debug, 22)
    graph = GraphBuilder('CESM/Data/Techmap', app, debug).build()
    chat = Chat(graph, 40, debug)
    app.set_chat(chat)
    app.mainloop()

if __name__ == '__main__':
    main()
    