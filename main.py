import os
from langchain_openai import ChatOpenAI
from webpage_screenshot import WebpageScreenshot
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
load_dotenv()

class UserStory(BaseModel):
    user_story: str = Field(
        description="User Story as it is without any modification")
    status: bool = Field(
        description="User Story pass or fail. Pass denoted by true and Fail denoted by false")
    reason: str = Field(
        description="Reason for the status of the User Story")

class UserStories(BaseModel):
    user_stories: List[UserStory]


def run_user_stories():
    webpage_content = WebpageScreenshot("https://en.wikipedia.org/wiki/Main_Page")

    parser = JsonOutputParser(pydantic_object=UserStories)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a QA Engineer at a software company. "
            "Your task is to perform browser tests to check if the website meets the given user stories. "
            "If the user story is met return the user story followed by true/false based on if it is satisfied or not."
            "If a user story is not satisfied return the reason for failure."
            f"Format Instructions: {parser.get_format_instructions()}"
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": (
                    "User Story: As a User, I should be able to see Today's feature article."
                    "User Story: As a User, I should be able to see 'In the news' section."
                    "User Story: As a User, I should be able to see and option to Sign Up."
                    "User Story: As a User, I should be able to see an option to Log In"
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{webpage_content}",
                    "detail": "auto",
                }}
            ]
        )
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = (prompt | llm)

    response = chain.invoke(input={})
    print("Response: ", response)

run_user_stories()
