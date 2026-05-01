import os
import tempfile
from IPython.display import display, Image
from sys import platform
from dotenv import load_dotenv
import getpass
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

#==================setup=====================
load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

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
        print(result.stdout + '\n' + result.stderr)
        return result.stdout + "\n" + result.stderr
    else:
        stream = os.popen(command)
        output = stream.read()
        print(output)
        return output


search_tool = TavilySearch(
    api_key=os.environ.get("TAVILY_API_KEY"),
)
tools = [execute_command, search_tool]
model = model.bind_tools(tools)


# =============Nodes================
def agent(state: AgentState) -> AgentState:
    messages = state["messages"]

    # Check if the last message was a ToolMessage.
    # If it was, the AI needs to process the result
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

        user_input = input(prompt)
        if user_input.lower() == "e":
            return {"messages": [HumanMessage(content="e")]}

        user_message = HumanMessage(content=user_input)
        sys_message = SystemMessage(
            content="You are an OS agent. If you need info, use search_tool. "
                    "If you need to act, use execute_command. "
                    "see the problems that might happen after execution and deal with it"
                    f"User OS: {get_os_details()}"
        )
        # Combine history and new message
        response = model.invoke([sys_message] + messages + [user_message])
        # Return both the user message and the AI response to the state
        return {"messages": [user_message, response]}

    # Print the AI's thoughts if there is text content
    if response.content:
        print(f"\nAI: {response.content}\n")

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
            print(f"WARNING: AI wants to run: {last_message.tool_calls[0]['args'].get('command')}")
            choice = input("Allow execution? (y/n): ").lower()
            if choice not in ["y", "yes"]:
                # Inject a failure message into state so the AI knows it was denied
                state["messages"].append(HumanMessage(content="Permission denied by user."))
                return "agent"

        return "tools"

    # If the last message was a ToolMessage, always go back to agent to summarize
    if isinstance(last_message, ToolMessage):
        return "agent"

    # If it's just a normal AI message with no tool calls, wait for next user input
    return "agent"

#===========Graph================
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


#===========app=================


def run():
    """Entry point for the terminal command"""
    try:
        app.invoke({"messages": []})
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    app = graph.compile()
    run()