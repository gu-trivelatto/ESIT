from abc import ABC
import customtkinter

class ToplevelWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        # Extract label, buttons, and callback from kwargs
        self.callback = kwargs.pop('callback', None)

        super().__init__(*args, **kwargs)
        self.geometry("400x300")
        
        # Create the scrollable frame and place it inside the window
        self.frame = customtkinter.CTkScrollableFrame(self, width=400, height=300)
        self.frame.pack(fill="both", expand=True)
        
        # Configure the grid layout for the scrollable frame
        self.frame.grid_columnconfigure(0, weight=1)

        # Add widgets to the scrollable frame
        self.label = customtkinter.CTkLabel(self.frame, text="Pop-up label")
        self.label.grid(row=0, column=0, padx=10, pady=10)

        self.button = customtkinter.CTkButton(self.frame, text='Button text', command=lambda: self.button_clicked('button_clicked'))
        self.button.grid(row=1, column=0, padx=10, pady=10)

    def button_clicked(self, text) -> None:
        if self.callback:
            self.callback(text)
        self.destroy()

class TestFunction(ABC):
    def __init__(self, app):
        self.app = app
        self.toplevel_window = None
        
    def main_runner(self):
        print('Running something')
        c = 3 ** 4
        print(c)
        print('Did something else')
        
        self.invoke()
        
        print('Finishing execution')
        
        return

    def invoke(self) -> str:
        def on_button_click(selected_value):
            print(f"Selected button: {selected_value}")
            # Do something with the selected value

        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            kwargs = {
                'callback': on_button_click
            }
            self.toplevel_window = ToplevelWindow(self.app, **kwargs)  # create window if it's None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it