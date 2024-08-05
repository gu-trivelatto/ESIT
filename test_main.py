from tkinter import *
import customtkinter
from test_function import TestFunction

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        customtkinter.set_appearance_mode('dark')
        
        self.title("ESMChat")
        self.geometry("400x400")

        self.button = customtkinter.CTkButton(master=self, command=self.run_test, text="Test", height=80)
        self.button.pack(padx=20, pady=20)
        
    def run_test(self):
        self.test_function.main_runner()
        
    def set_test_function(self, function):
        self.test_function = function
        
if __name__ == '__main__':
    app = App()
    test_function = TestFunction(app)
    app.set_test_function(test_function)
    app.mainloop()
    