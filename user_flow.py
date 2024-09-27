import platform
import re
import base64
import asyncio
from typing import List, Optional, TypedDict
from playwright.async_api import async_playwright
from playwright.async_api import Page
from langgraph.graph import END, START, StateGraph
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain as chain_decorator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain import hub
from dotenv import load_dotenv
load_dotenv()


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    # The bounding boxes from the browser annotation function
    bboxes: List[BBox]
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool


async def click(state: AgentState):
    # - Click [Numerical_Label]
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    # TODO: In the paper, they automatically parse any downloaded PDFs
    # We could add something similar here as well and generally
    # improve response format.
    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")

    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."


with open("mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            print("Exception")
    screenshot = await page.screenshot(full_page=True)
    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }


# Agent Definition
async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    pattern = r"Thought: (.+)\n"
    match = re.search(pattern, text)
    thought_content = ""
    if match:
        thought_content = match.group(1)
        print(thought_content)
    else:
        print("Thought content not found.")

    pattern = r"Completed User Story: (.+)\n"
    match = re.search(pattern, text)
    completed_user_story_content = ""
    if match:
        completed_user_story_content = match.group(1)
        print(completed_user_story_content)
    else:
        print("Completed User Story content not found.")

    pattern = r"Pending User Story: (.+)\n"
    match = re.search(pattern, text)
    pending_user_story_content = ""
    if match:
        pending_user_story_content = match.group(1)
        print(pending_user_story_content)
    else:
        print("Pending User Story content not found.")

    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input, "thought": thought_content,
            "completed_user_story": completed_user_story_content, "pending_user_story": pending_user_story_content }


prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(
        prompt=[
            PromptTemplate(
                input_variables=[],
                template=(
                    "Imagine you are a QA robot browsing the web, just like a human QA Engineer. Now you need to complete a task. "
                    "In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. "
                    "This screenshot will\nfeature Numerical Labels placed in the TOP LEFT corner of each Web Element. "
                    "Carefully analyze the visual\ninformation to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow\nthe guidelines and choose one of the following actions:\n\n"
                    "1. Click a Web Element.\n2. Delete existing content in a textbox and then type content.\n3. Scroll up or down.\n4. Wait \n5. Go back\n7. Return to google to start over.\n8. Respond with the final answer\n\n"
                    "Correspondingly, Action should STRICTLY follow the format:\n\n- Click [Numerical_Label] \n- Type [Numerical_Label]; [Content] \n- Scroll [Numerical_Label or WINDOW]; [up or down] \n- Wait \n- GoBack\n- Google\n- ANSWER; [content]\n\n"
                    "Key Guidelines You MUST follow:\n\n* Action guidelines *\n"
                    "1) Execute only one action per iteration.\n"
                    "2) When clicking or typing, ensure to select the correct bounding box.\n"
                    "3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n\n"
                    "* Web Browsing Guidelines *\n"
                    "1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages\n"
                    "2) Select strategically to minimize time wasted.\n\n"
                    "Your reply should strictly follow the format:\n\n"
                    "Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}\n"
                    "Completed User Story: {{Portion of the User Story completed by executing the action.}}\n"
                    "Pending User Story: {{Portion of the User Story that is left to test after executing the action.}}\n"
                    "Action: {{One Action format you choose}}\n"
                    "Then the User will provide:\nObservation: {{A labeled screenshot Given by User}}\n\n"
                    "Your ultimate goal is to perform browser tests to check if the website meets the given user stories. "
                    "If there is no pending user story to test and ff the user story is met in total return the user story followed by true/false based on if it is satisfied or not."
                    "If a user story is not satisfied return the reason for failure."
                )
            )]),
    MessagesPlaceholder(variable_name='scratchpad', optional=True),
    HumanMessagePromptTemplate(
        prompt=[
            ImagePromptTemplate(
                input_variables=['img'],
                template={'url': 'data:image/png;base64,{img}'}),
            PromptTemplate(input_variables=[
                'bbox_descriptions'], template='{bbox_descriptions}'),
            PromptTemplate(input_variables=['input'], template='{input}')
        ])])
# prompt = hub.pull("wfh/web-voyager")
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)


def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\nAI Thought: {state['prediction']['thought']}"
    txt += f"\nCompleted User Story: {state['prediction']['completed_user_story']}"
    txt += f"\nPending User Story: {state['prediction']['pending_user_story']}"
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}


graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Wait": wait,
    "Scroll": scroll,
    "GoBack": go_back,
    "Google": to_google,
}


for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        # The lambda ensures the function's string output is mapped to the "observation"
        # key in the AgentState
        RunnableLambda(tool) | (lambda observation: {
            "observation": observation}),
    )
    # Always return to the agent (by means of the update-scratchpad node)
    graph_builder.add_edge(node_name, "update_scratchpad")


def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action


graph_builder.add_conditional_edges("agent", select_tool)

graph = graph_builder.compile()


async def AsyncWebpageBroswer(url):
    browser = await async_playwright().start()
    # We will set headless=False so we can watch the agent navigate the web.
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    _ = await page.goto(url)


async def call_agent(question: str, page, max_steps: int = 50):
    browser = await async_playwright().start()
    # We will set headless=False so we can watch the agent navigate the web.
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    _ = await page.goto("https://www.theverge.com/")
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    async for event in event_stream:
        # We'll display an event stream here
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        print("\n".join(steps))
        if "ANSWER" in action:
            final_answer = action_input[0]
            break
    return final_answer

res = asyncio.run(call_agent(
    "User Story: As a User, when I click on an 'Article' headline on the Top Stories I should be taken to the particular story page.", ""))
print(f"Final response: {res}")
