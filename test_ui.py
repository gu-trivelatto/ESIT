from tkinter import *
from datetime import datetime

root = Tk()
root.config(bg="lightblue")

canvas = Canvas(root, width=200, height=200,bg="white")
canvas.grid(row=0,column=0,columnspan=2)

bubbles = []

class BotBubble:
    def __init__(self,master,message=""):
        self.master = master
        self.frame = Frame(master,bg="light grey")
        self.i = self.master.create_window(90,160,window=self.frame)
        Label(self.frame,text=datetime.now().strftime("%Y-%m-%d %H:%m"),font=("Helvetica", 7),bg="light grey").grid(row=0,column=0,sticky="w",padx=5)
        Label(self.frame, text=message,font=("Helvetica", 9),bg="light grey").grid(row=1, column=0,sticky="w",padx=5,pady=3)
        root.update_idletasks()
        self.master.create_polygon(self.draw_triangle(self.i), fill="light grey", outline="light grey")

    def draw_triangle(self,widget):
        x1, y1, x2, y2 = self.master.bbox(widget)
        return x1, y2 - 10, x1 - 15, y2 + 10, x1, y2

def send_message():
    if bubbles:
        canvas.move(ALL, 0, -65)
    a = BotBubble(canvas,message=entry.get())
    bubbles.append(a)

entry = Entry(root,width=26)
entry.grid(row=1,column=0)
Button(root,text="Send",command=send_message).grid(row=1,column=1)
root.mainloop()