import os
import traceback
from sys import platform
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_tavily import TavilySearch
from typing import TypedDict, Annotated
from langchain_openrouter import ChatOpenRouter
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from os_detection import get_os_details
import subprocess
from langchain.tools import tool
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


load_dotenv()
console = Console()

model = ChatOpenRouter(
    model="poolside/laguna-xs.2:free",
    temperature=0
)

#===============tools==============
@tool
def execute_command(command: str) -> str:
    """executes a command and returns the output of the command
        :param command: a string representing the command to be executed
        :returns the output of the command
    """
    #we use something generic for all OSs
    if platform.startswith("win"):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        console.print(result.stdout + '\n' + result.stderr)
        return result.stdout + "\n" + result.stderr
    else:
        stream = os.popen(command)
        output = stream.read()
        console.print(output)
        return output





# =============Nodes================
def agent(state: AgentState) -> AgentState:
    messages = state["messages"]

    # Check if the last message was a ToolMessage.
    if messages and isinstance(messages[-1], ToolMessage):
        sys_message = SystemMessage(
            content=f"Analyze the tool output and report back to the user. OS: {get_os_details()}")
        response = model.invoke([sys_message] + messages)
    else: #ask user for input
        if not messages:
            prompt = ("Hello! I'm your terminal friend. I can help you execute commands,"
                      " I analyse problems that might appear,"
                      " and even search the web for solutions!"
                      " how can I help you today?\n")
        else:
            prompt = "What's the next step? (type 'e' to exit)\n"

        user_input = Prompt.ask(prompt)
        if user_input.lower() == "e":
            return {"messages": [HumanMessage(content="e")]}

        user_message = HumanMessage(content=user_input)
        sys_message = SystemMessage(
            content="You are an OS agent. If you need info, use search_tool. "
                    "If you need to act, use execute_command. "
                    "see the problems that might happen after execution and deal with it"
                    f"User OS: {get_os_details()}"
        )

        response = model.invoke([sys_message] + messages + [user_message])
        console.print(Panel(response.content, title="[bold blue]AI Response[/bold blue]", border_style="blue"))
        # Return both the user message and the AI response to the state
        return {"messages": [user_message, response]}

    # Print the AI's thoughts if there is text content
    if response.content:
        console.print(Panel(f"\n{response.content}\n", title="[bold green]AI Analysis[/bold green]", border_style="green"))

    return {"messages": [response]}


# ================conditional edges================
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # Handle Exit
    if isinstance(last_message, HumanMessage) and last_message.content == "e":
        return "end"

    # If AI wants to call a tool
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Permission logic for command execution
        if any(tc['name'] == "execute_command" for tc in last_message.tool_calls):
            console.print(f"[bold red]WARNING:[/bold red] AI wants to run: {last_message.tool_calls[0]['args'].get('command')}")
            choice = Prompt.ask("Do you give permission? (y/n)", choices=["y", "n", "yes", "no"])
            if choice.lower() in ["y", "yes"]:
                return "tools"  # go to the tools to execute the command
            elif choice.lower() in ["n", "no"]:
                console.print("[yellow]Command was not executed.[/yellow]")
                state["messages"].append(HumanMessage("Permission to execute the command was denied by the user."))
                return "agent"  # go back to the agent to ask

        return "tools"

    # If the last message was a ToolMessage, always go back to agent to summarize
    if isinstance(last_message, ToolMessage):
        return "agent"

    # If it's just a normal AI message with no tool calls, wait for next user input
    return "agent"





def welcome_user():
    welcome_text = ("Hi there! I am your Terminal friend!\n"
                    "I can help you execute commands and fix problems that might appear in the way.\n"
                    "I can read the output of commands I execute.\n"
                    "And I can even search the web for solutions.\n")
    console.print(Panel(welcome_text, title="[bold magenta]Welcome to Terminal Friend![/bold magenta]", border_style="magenta"))

    info_text = ("I work with open router for the AI model and Tavily search for searching.\n"
                 "For both, you can get an API key from official sites:\n"
                 "- Open Router: https://openrouter.ai/\n"
                 "- Tavily: https://www.tavily.com/\n"
                 "You can even get free models from open router, and free 1000 API credits/month from Tavily!")
    console.print(Panel(info_text, title="[bold cyan]API Information[/bold cyan]", border_style="cyan"))

    choice = Prompt.ask("Do you want to add the API keys manually in the .env file, or enter it here and we add it automatically?", choices=["a", "m"], default="a")

    if choice == "a":
        open_router_api = Prompt.ask("Please enter your Open Router API key")
        tavily_search_api = ''
        choice = Prompt.ask("Do you want to use the search tool?", choices=["y", "n", "yes", "no"])
        if choice.lower() in ["y", "yes"]:
            tavily_search_api = Prompt.ask("Please enter your Tavily search API key")

        with open('../.env', 'a') as f:
            f.write(f"OPENROUTER_API_KEY={open_router_api}\n")
            f.write(f"TAVILY_API_KEY={tavily_search_api}\n")
    elif choice == "m":
        console.print("[green]Great! Please write your keys inside the .env file manually.[/green]")
        Prompt.ask("Press enter when you are done")


def is_first_time():
    with open('log.txt', 'r') as f:
        if not f.read():
            return True
        else:
            return False

def log(content : str) -> None:
    with open('log.txt', 'a') as f:
        f.write(content + '\n')

def run():
    """Entry point for the terminal command"""

    if is_first_time():
        welcome_user()
        log("user used program for the first time.")

    # ==================setup=====================

    search_tool = TavilySearch(
        api_key=os.environ.get("TAVILY_API_KEY"),
    )
    tools = [execute_command, search_tool]
    global model
    model = model.bind_tools(tools)

    # ===========Graph================
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools=tools))
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "agent",
            "tools": "tools",
            "end": END
        }
    )

    # ===========app=================
    app = graph.compile()


    try:
        app.invoke({"messages": []})
    except KeyboardInterrupt:
        console.print("\n[bold red]Goodbye![/bold red]")
    except:
        log(traceback.format_exc())

if __name__ == "__main__":
    run()