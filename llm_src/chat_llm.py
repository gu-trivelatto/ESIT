import os
import yaml
import llm_src.agents as agents
import llm_src.routers as routers
import llm_src.printers as printers

from abc import ABC
from langchain_groq import ChatGroq
from llm_src.state import GraphState
from langgraph.graph import END, StateGraph
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

with open('secrets.yml', 'r') as f:
    secrets = yaml.load(f, Loader=yaml.SafeLoader)

os.environ["GROQ_API_KEY"] = secrets['groq'][0]
chat_model = ChatGroq(
            model="llama3-70b-8192",
        )
json_model = ChatGroq(
            model="llama3-70b-8192",
        ).bind(response_format={"type": "json_object"})

debug = True

class RAGEmbedder(ABC):
    def __init__(self):
        # Load the data that will be used by the retriever
        loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
        docs = loader.load()

        # Set the embedding model
        embeddings = OllamaEmbeddings(model="llama3")

        # Split the data and vectorize it
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)

        # Define a chain to gather data and a retriever
        self.retriever = vector.as_retriever()
    
    def execute(self, query):
        return self.retriever.invoke(query)
    
class WebSearchTool(ABC):
    def __init__(self):
        os.environ["TAVILY_API_KEY"] = secrets['tavily'][0]
        self.web_search_tool = TavilySearchResults()
    
    def execute(self, query):
        return self.web_search_tool.invoke({"query": query})
    
retriever = RAGEmbedder().retriever
web_tool = WebSearchTool().web_search_tool

class GraphBuilder(ABC):
    def __init__(self, debug):
        self.debug = debug
    
    def type_identifier(self, state: GraphState) -> GraphState:
        return agents.TypeIdentifier(chat_model, json_model, state, self.debug).execute()

    def es_tool_selector(self, state: GraphState) -> GraphState:
        return agents.ESToolSelector(chat_model, json_model, state, self.debug).execute()

    def model_selector(self, state: GraphState) -> GraphState:
        return agents.ModelSelector(chat_model, json_model, state, self.debug).execute()

    def mixed(self, state: GraphState) -> GraphState:
        return agents.Mixed(chat_model, json_model, state, self.debug).execute()

    def tool_selector(self, state: GraphState) -> GraphState:
        return agents.ToolSelector(chat_model, json_model, state, self.debug).execute()

    def research_info_rag(self, state: GraphState) -> GraphState:
        return agents.ResearchInfoRAG(chat_model, json_model, retriever, web_tool, state, self.debug).execute()

    def research_info_web(self, state: GraphState) -> GraphState:
        return agents.ResearchInfoWeb(chat_model, json_model, retriever, web_tool, state, self.debug).execute()

    def calculator(self, state: GraphState) -> GraphState:
        return agents.Calculator(chat_model, json_model, state, self.debug).execute()

    def date_getter(self, state: GraphState) -> GraphState:
        return agents.DateGetter(chat_model, json_model, state, self.debug).execute()

    def param_selector(self, state: GraphState) -> GraphState:
        return agents.ParamsIdentifier(chat_model, json_model, state, self.debug).execute()

    def scenario_selector(self, state: GraphState) -> GraphState:
        return agents.ScenarioIdentifier(chat_model, json_model, state, self.debug).execute()

    def plot_selector(self, state: GraphState) -> GraphState:
        return agents.PlotIdentifier(chat_model, json_model, state, self.debug).execute()

    def model_modifier(self, state: GraphState) -> GraphState:
        return agents.ModelModifier(chat_model, json_model, state, self.debug).execute()

    def sim_runner(self, state: GraphState) -> GraphState:
        return agents.SimRunner(chat_model, json_model, state, self.debug).execute()

    def plotter(self, state: GraphState) -> GraphState:
        return agents.Plotter(chat_model, json_model, state, self.debug).execute()

    def output_generator(self, state: GraphState) -> GraphState:
        return agents.OutputGenerator(chat_model, json_model, state, self.debug).execute()

    def context_analyzer(self, state: GraphState) -> GraphState:
        return agents.ContextAnalyzer(chat_model, json_model, state, self.debug).execute()

    def empty_node(self, state: GraphState) -> GraphState:
        return agents.EmptyNode(chat_model, json_model, state, self.debug).execute()

    def state_printer(self, state: GraphState) -> None:
        return printers.StatePrinter(state, self.debug).execute()

    def final_answer_printer(self, state: GraphState) -> None:
        return printers.FinalAnswerPrinter(state, self.debug).execute()

    def route_to_type(self, state: GraphState) -> str:
        return routers.RouteToType(state, self.debug).execute()

    def route_from_mix(self, state: GraphState) -> str:
        return routers.RouteFromMix(state, self.debug).execute()

    def validate_selected_model(self, state: GraphState) -> str:
        return routers.ValidateSelectedModel(state, self.debug).execute()

    def route_to_es_tool(self, state: GraphState) -> str:
        return routers.RouteToESTool(state, self.debug).execute()

    def selection_validator(self, state: GraphState) -> str:
        return routers.SelectionValidator(state, self.debug).execute()

    def route_to_iterate(self, state: GraphState) -> str:
        return routers.RouteToIterate(state, self.debug).execute()

    def route_to_tool(self, state: GraphState) -> str:
        return routers.RouteToTool(state, self.debug).execute()

    def build(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("type_identifier", self.type_identifier)
        workflow.add_node("es_tool_selector", self.es_tool_selector)
        workflow.add_node("model_selector", self.model_selector)
        workflow.add_node("validated_model", self.empty_node)
        workflow.add_node("mixed", self.mixed)
        workflow.add_node("tool_selector", self.tool_selector)
        workflow.add_node("research_info_rag", self.research_info_rag) # RAG search
        workflow.add_node("research_info_web", self.research_info_web) # web search
        workflow.add_node("state_printer", self.state_printer)
        workflow.add_node("calculator", self.calculator)
        workflow.add_node("date_getter", self.date_getter)
        workflow.add_node("inter_node", self.empty_node)
        workflow.add_node("param_selector", self.param_selector)
        workflow.add_node("scenario_selector", self.scenario_selector)
        workflow.add_node("plot_selector", self.plot_selector)
        workflow.add_node("model_modifier", self.model_modifier)
        workflow.add_node("sim_runner", self.sim_runner)
        workflow.add_node("plotter", self.plotter)
        workflow.add_node("output_generator", self.output_generator)
        workflow.add_node("context_analyzer", self.context_analyzer)
        workflow.add_node("final_answer_printer", self.final_answer_printer)

        workflow.set_entry_point("date_getter")
        workflow.add_edge("date_getter", "type_identifier")
        workflow.add_conditional_edges(
            "type_identifier",
            self.route_to_type,
            {
                "general": "context_analyzer",
                "energy_system": "es_tool_selector",
                "mixed": "mixed",
            }
        )

        workflow.add_conditional_edges(
            "mixed",
            self.route_from_mix,
            {
                "complete_data": "es_tool_selector",
                "needs_data": "context_analyzer"
            }
        )

        workflow.add_conditional_edges(
            "es_tool_selector",
            self.validate_selected_model,
            {
                "select_model": "model_selector",
                "model_is_valid": "validated_model"
            }
        )

        workflow.add_conditional_edges(
            "validated_model",
            self.route_to_es_tool,
            {
                "data_plotter": "scenario_selector",
                "sim_runner": "scenario_selector",
                "model_modifier": "param_selector"
            }
        )

        workflow.add_conditional_edges(
            "model_selector",
            self.route_to_es_tool,
            {
                "data_plotter": "scenario_selector",
                "sim_runner": "scenario_selector",
                "model_modifier": "param_selector"
            }
        )

        workflow.add_conditional_edges(
            "scenario_selector",
            self.route_to_es_tool,
            {
                "data_plotter": "plot_selector",
                "sim_runner": "inter_node"
            }
        )

        workflow.add_conditional_edges(
            "param_selector",
            self.selection_validator,
            {
                "model_modifier": "model_modifier",
                "end_not_valid": "output_generator"
            }
        )

        workflow.add_conditional_edges(
            "plot_selector",
            self.selection_validator,
            {
                "data_plotter": "plotter",
                "end_not_valid": "output_generator"
            }
        )

        workflow.add_conditional_edges(
            "inter_node",
            self.selection_validator,
            {
                "sim_runner": "sim_runner",
                "end_not_valid": "output_generator"
            }
        )
        workflow.add_edge("model_modifier", "output_generator")
        workflow.add_edge("plotter", "output_generator")
        workflow.add_edge("sim_runner", "output_generator")

        workflow.add_conditional_edges(
            "context_analyzer",
            self.route_to_iterate,
            {
                "ready_to_answer": "output_generator",
                "need_context": "tool_selector",
            },
        )

        workflow.add_conditional_edges(
            "tool_selector",
            self.route_to_tool,
            {
                "RAG_retriever": "research_info_rag",
                "web_search": "research_info_web",
                "calculator": "calculator",
            },
        )
        workflow.add_edge("research_info_rag", "state_printer")
        workflow.add_edge("research_info_web", "state_printer")
        workflow.add_edge("calculator", "state_printer")

        workflow.add_conditional_edges(
            "state_printer",
            self.route_to_type,
            {
                "general": "context_analyzer",
                "mixed": "mixed",
            }
        )

        workflow.add_edge("output_generator", "final_answer_printer")
        workflow.add_edge("final_answer_printer", END)

        self.app = workflow.compile()
        return self.app