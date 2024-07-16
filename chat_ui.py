import os
import pickle
from abc import ABC
from langgraph.graph import StateGraph

#query = 'If I pay half the age of Tom Jobim plus the height of the Empire State for a car, how much I\'ve paid?'
#query = 'What is 10 to the power of 0.4?'
#query = 'What is the temperature and humidity in Darmstadt right now? And also, what time is it?'
#query = 'Modify the parameter X to 24 for me please'
#query = 'What are some of the most important things that happened today in past years?'
#query = 'What day is today?'
#query = 'How can LangSmith help in my project?'
#query = 'I am always coming but never arrive. What am I?'
#query = 'Change the lifetime of wind power plants to 25 years please'
#query = 'Divide the height of the Burj Khalifa by Ronaldinho Gaucho\'s age, then add the current temperature in Paris (in Celsius)'
#query = 'What are good famous and more casual board games that can be played by two players?'
#query = 'Divide the number of visitors that the Eiffel tower receives yearly by the number of cars in the city of São Paulo, Brazil'
#query = 'Change the technical lifetime of wind power plants to be the age of Olaf Scholz'
#query = 'Modify the lifetime of wind power plants to be the same value as the price of one liter of Coca Cola in Brazil.'
#query = 'Modify the investment cost power of the Biomass CHP to be the number of years michael jackson has been dead to the power of 1.5'

query = 'Is my civic faster than a Ferrari?'
#query = 'Of course, my car is a 1998 Honda Civic 1.6'
#query = 'Well, what would I need to make it faster than a ferrari?'
#query = 'What would be the risk to the engine when upgrading it to reach about 500hp?'
#query = 'Can you suggest any mechanic near the Frankfurt area where I could ask about modifications?'
#query = 'Okay, more specifically it can be in Darmstadt'
#query = 'Can you tell me their address?'
#query = 'None of them exist in the map, where did you get this information from?'
#query = 'Can you use the built-in web search tool to find actual mechanics?'

query = 'What is the current temperature in Darmstadt?'

class Chat(ABC):
    def __init__(self, app: StateGraph):
        self.app = app
        os.remove("chat_history.pkl")

    def invoke(self, input = query) -> None:
        # run the agent
        try:
            with open("chat_history.pkl", "rb") as f:
                history = pickle.load(f)
        except:
            history = []
        history.append({"role": "user", "content": input})

        inputs = {"initial_query": history, "next_query": '', "num_steps": 0, "context": [], "history": history}
        for output in self.app.stream(inputs, {"recursion_limit": 50}):
            for key, value in output.items():
                print(f"Finished running <{key}> \n")