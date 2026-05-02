import os
import traceback
from pathlib import Path
from sys import platform
from dotenv import load_dotenv, set_key
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from firecrawl import Firecrawl
from typing import TypedDict, Annotated
from langchain_openrouter import ChatOpenRouter
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from .os_detection import get_os_details
import subprocess
from langchain.tools import tool
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

#to save the context size for the llm
def add_message(history, new_message, max_depth=10):
    history.append(new_message)
    # Keep the system prompt (index 0) + the last N messages
    if len(history) > max_depth:
        history = [history[0]] + history[-(max_depth-1):]
    return history

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


console = Console()
model = ''


# ===============tools==============
@tool
def search_web(url: str) -> str:
    """fetches and crawls a URL using FireCrawl returning the content
        :param url: URL to fetch and crawl
        :returns the content from the URL
    """
    try:
        client = Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY"))
        response = client.scrape(url, only_main_content=True)
        if response and 'content' in response:
            return response['content']
        else:
            return "No content found"
    except Exception as e:
        return f"Error crawling URL: {str(e)}"


@tool
def execute_command(command: str) -> str:
    """executes a command and returns the output of the command
        :param command: a string representing the command to be executed
        :returns the output of the command
    """
    # Fix: Move permission logic inside the tool to prevent LangGraph state desync
    console.print(f"\n[bold red]WARNING:[/bold red] AI wants to run: [cyan]{command}[/cyan]")
    choice = Prompt.ask("Do you give permission? (y/n)", choices=["y", "n", "yes", "no"])

    if choice.lower() not in ["y", "yes"]:
        console.print("[yellow]Command execution blocked by user.[/yellow]")
        return "User denied permission to execute this command. Acknowledge this and ask the user what to do next."

    if platform.startswith("win"):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout + '\n' + result.stderr
        console.print(output)
        return output
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
        # tell AI to anaylze the tool output and don't loop over tools forever
        sys_message = SystemMessage(
            content=f"Analyze the tool output and answer the user. If you have the information you need, STOP calling tools and give the final answer. User OS: {get_os_details()}"
        )
        response = model.invoke([sys_message] + messages)
    else:  # ask user for input
        if not messages:
            prompt = ("Hello! I'm your terminal friend.\nI can help you execute commands,\n"
                      "I analyse problems that might appear,\n"
                      "and even search the web for solutions!\n"
                      "how can I help you today?\n")
            console.print(Panel(prompt.strip(), title="[bold blue]Terminal Friend[/bold blue]", border_style="blue"))
            user_input = Prompt.ask("")
        else:
            prompt = "What's the next step? (type 'e' to exit)\n"
            console.print(Panel(prompt.strip(), title="[bold blue]Next Step[/bold blue]", border_style="blue"))
            user_input = Prompt.ask("")

        #user exits
        if user_input.lower() == "e":
            return {"messages": [HumanMessage(content="e")]}

        #give the user's input to the AI along with a system message
        user_message = HumanMessage(content=user_input)
        sys_message = SystemMessage(
            content="You are an OS agent. If you need info, use search_web to crawl URLs if it exists in your bonded tools."
                    "If you need to act, use execute_command. "
                    "see the problems that might happen after execution and deal with it. "
                    f"User OS: {get_os_details()}"
        )

        response = model.invoke([sys_message] + messages + [user_message])
        console.print(Panel(response.content, title="[bold blue]AI Response[/bold blue]", border_style="blue"))
        return {"messages": [user_message, response]}

    if response.content:
        console.print(
            Panel(f"\n{response.content}\n", title="[bold green]AI Analysis[/bold green]", border_style="green"))

    return {"messages": [response]}


# ================conditional edges================
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # Handle Exit
    if isinstance(last_message, HumanMessage) and last_message.content == "e":
        return "end"

    # If AI wants to call a tool
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # If the last message was a ToolMessage, or a normal AI message, go back to agent
    return "agent"


# ================setup & utilities================
def welcome_user():
    welcome_text = ("Hi there! I am your Terminal friend!\n"
                    "I can help you execute commands and fix problems that might appear in the way.\n"
                    "I can read the output of commands I execute.\n"
                    "And I can even crawl websites for information using FireCrawl.\n")
    console.print(
        Panel(welcome_text, title="[bold magenta]Welcome to Terminal Friend![/bold magenta]", border_style="magenta"))

    info_text = ("I work with open router for the AI model and FireCrawl for web scraping.\n"
                 "For both, you can get an API key from official sites:\n"
                 "- Open Router: https://openrouter.ai/\n"
                 "- FireCrawl: https://www.firecrawl.dev/\n"
                 "You can get free models from open router, and use FireCrawl's free tier!")
    console.print(Panel(info_text, title="[bold cyan]API Information[/bold cyan]", border_style="cyan"))

    choice = Prompt.ask(
        "Do you want to add the API keys manually in the .env file, or enter it here and we add it automatically?",
        choices=["a", "m"], default="a")

    if choice == "a":
        open_router_api = Prompt.ask("Please enter your Open Router API key")
        firecrawl_api = ''
        choice_web = Prompt.ask("Do you want to use the web crawling tool?", choices=["y", "n", "yes", "no"])
        if choice_web.lower() in ["y", "yes"]:
            firecrawl_api = Prompt.ask("Please enter your FireCrawl API key")

        env_path = Path(__file__).parent / ".env"
        set_key(env_path,"OPENROUTER_API_KEY", open_router_api)
        set_key(env_path, "FIRECRAWL_API_KEY", firecrawl_api)

    elif choice == "m":
        console.print("[green]Great! Please write your keys inside the .env file manually.[/green]")
        Prompt.ask("Press enter when you are done")


def is_first_time():
    if not os.path.exists('log.txt'):
        open('log.txt', 'w').close()
        return True

    with open('log.txt', 'r') as f:
        return not bool(f.read().strip())


def log(content: str) -> None:
    with open('log.txt', 'a') as f:
        f.write(content + '\n')


def run():
    """Entry point for the terminal command"""
    if is_first_time():
        welcome_user()
        log("user used program for the first time.")

    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    global model
    model = ChatOpenRouter(
        model="poolside/laguna-xs.2:free",
        temperature=0
    )
    # ==================setup=====================
    tools = [execute_command, search_web]

    if not os.environ.get("FIRECRAWL_API_KEY"):
        tools.pop()

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

    app = graph.compile()

    try:
        # Fix: Added a recursion limit. If the model loops 15 times, it will force-stop.
        app.invoke({"messages": []}, {"recursion_limit": 15})
    except KeyboardInterrupt:
        console.print("\n[bold red]Goodbye![/bold red]")
    except Exception as e:
        # Fix: Print the error to the console so it doesn't fail silently
        console.print(f"\n[bold red]Application Error:[/bold red] {e}")
        console.print("Check log.txt for full traceback.")
        log(traceback.format_exc())


if __name__ == "__main__":
    run()
