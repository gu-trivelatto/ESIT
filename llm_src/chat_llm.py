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
        return self.web_search_tool.invoke({"query": query, "max_results": 3})
    
retriever = RAGEmbedder().retriever
web_tool = WebSearchTool()

class GraphBuilder(ABC):
    def __init__(self, model_path, app, debug):
        self.base_model = f'{model_path}/DEModel.xlsx'
        self.mod_model = f'{model_path}/DEModel_modified.xlsx'
        self.debug = debug
        self.app = app
    
    # Agents (Nodes of the Graph)
    
    def date_getter(self, state: GraphState) -> GraphState:
        return agents.DateGetter(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def type_identifier(self, state: GraphState) -> GraphState:
        return agents.TypeIdentifier(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def mixed(self, state: GraphState) -> GraphState:
        return agents.Mixed(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def context_analyzer(self, state: GraphState) -> GraphState:
        return agents.ContextAnalyzer(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def tool_selector(self, state: GraphState) -> GraphState:
        return agents.ToolSelector(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def research_info_rag(self, state: GraphState) -> GraphState:
        return agents.ResearchInfoRAG(chat_model, json_model, retriever, web_tool, state, self.app, self.debug).execute()

    def research_info_web(self, state: GraphState) -> GraphState:
        return agents.ResearchInfoWeb(chat_model, json_model, retriever, web_tool, state, self.app, self.debug).execute()

    def calculator(self, state: GraphState) -> GraphState:
        return agents.Calculator(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def es_actions_analyzer(self, state: GraphState) -> GraphState:
        return agents.ESActionsAnalyzer(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()

    def run_model(self, state: GraphState) -> GraphState:
        return agents.RunModel(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def modify_model(self, state: GraphState) -> GraphState:
        return agents.ModifyModel(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def consult_model(self, state: GraphState) -> GraphState:
        return agents.ConsultModel(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def compare_model(self, state: GraphState) -> GraphState:
        return agents.CompareModel(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def plot_model(self, state: GraphState) -> GraphState:
        return agents.PlotModel(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    def actions_analyzer(self, state: GraphState) -> GraphState:
        return agents.ActionsAnalyzer(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()

    def output_generator(self, state: GraphState) -> GraphState:
        return agents.OutputGenerator(chat_model, json_model, self.base_model, self.mod_model, state, self.app, self.debug).execute()
    
    # Printers (nodes of the Graph)

    def state_printer(self, state: GraphState) -> None:
        return printers.StatePrinter(state, self.debug).execute()

    def final_answer_printer(self, state: GraphState) -> None:
        return printers.FinalAnswerPrinter(state, self.debug).execute()
    
    # Routers (conditional edges of the Graph)

    def type_router(self, state: GraphState) -> str:
        return routers.TypeRouter(state, self.debug).execute()

    def mixed_router(self, state: GraphState) -> str:
        return routers.MixedRouter(state, self.debug).execute()

    def es_action_router(self, state: GraphState) -> str:
        return routers.ESActionRouter(state, self.debug).execute()

    def context_router(self, state: GraphState) -> str:
        return routers.ContextRouter(state, self.debug).execute()

    def tool_router(self, state: GraphState) -> str:
        return routers.ToolRouter(state, self.debug).execute()
    
    ##### Build the Graph #####

    def build(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        ### Define the nodes ###
        workflow.add_node("date_getter", self.date_getter)
        workflow.add_node("type_identifier", self.type_identifier)
        workflow.add_node("es_actions_analyzer", self.es_actions_analyzer)
        workflow.add_node("context_analyzer", self.context_analyzer)
        workflow.add_node("mixed", self.mixed)
        workflow.add_node("tool_selector", self.tool_selector)
        workflow.add_node("research_info_rag", self.research_info_rag) # RAG search
        workflow.add_node("research_info_web", self.research_info_web) # web search
        workflow.add_node("calculator", self.calculator)
        workflow.add_node("run_model", self.run_model)
        workflow.add_node("modify_model", self.modify_model)
        workflow.add_node("consult_model", self.consult_model)
        workflow.add_node("compare_model", self.compare_model)
        workflow.add_node("plot_model", self.plot_model)
        workflow.add_node("actions_analyzer", self.actions_analyzer)
        workflow.add_node("output_generator", self.output_generator)
        workflow.add_node("es_state_printer", self.state_printer)
        workflow.add_node("context_state_printer", self.state_printer)
        workflow.add_node("final_answer_printer", self.final_answer_printer)

        ### Define the graph topography ###
        
        # Entry and query type routing
        workflow.set_entry_point("date_getter")
        workflow.add_edge("date_getter", "type_identifier")
        workflow.add_conditional_edges(
            "type_identifier",
            self.type_router,
            {
                "general": "context_analyzer",
                "energy_system": "es_actions_analyzer",
                "mixed": "mixed",
            }
        )

        # Mixed query routing
        workflow.add_conditional_edges(
            "mixed",
            self.mixed_router,
            {
                "complete_data": "es_actions_analyzer",
                "needs_data": "context_analyzer"
            }
        )

        # Energy System branch
        workflow.add_conditional_edges(
            "es_actions_analyzer",
            self.es_action_router,
            {
                "run": "run_model",
                "modify": "modify_model",
                "consult": "consult_model",
                "compare": "compare_model",
                "plot": "plot_model",
                "no_action": "output_generator"
            }
        )
        workflow.add_edge("run_model", "es_state_printer")
        workflow.add_edge("modify_model", "es_state_printer")
        workflow.add_edge("consult_model", "es_state_printer")
        workflow.add_edge("compare_model", "es_state_printer")
        workflow.add_edge("plot_model", "es_state_printer")
        workflow.add_edge("es_state_printer", "actions_analyzer")
        
        workflow.add_conditional_edges(
            "actions_analyzer",
            self.context_router,
            {
                "ready_to_answer": "output_generator",
                "need_context": "es_actions_analyzer",
            },
        )

        # General branch
        workflow.add_conditional_edges(
            "context_analyzer",
            self.context_router,
            {
                "ready_to_answer": "output_generator",
                "need_context": "tool_selector",
            },
        )

        workflow.add_conditional_edges(
            "tool_selector",
            self.tool_router,
            {
                "RAG_retriever": "research_info_rag",
                "web_search": "research_info_web",
                "calculator": "calculator",
                "user_input": "output_generator"
            },
        )
        workflow.add_edge("research_info_rag", "context_state_printer")
        workflow.add_edge("research_info_web", "context_state_printer")
        workflow.add_edge("calculator", "context_state_printer")

        workflow.add_conditional_edges(
            "context_state_printer",
            self.type_router,
            {
                "general": "context_analyzer",
                "mixed": "mixed",
            }
        )

        # Final steps (generate output and print the final answer)
        workflow.add_edge("output_generator", "final_answer_printer")
        workflow.add_edge("final_answer_printer", END)

        return workflow.compile()