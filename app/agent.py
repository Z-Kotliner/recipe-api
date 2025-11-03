import warnings

from github import Github, Auth
from pydantic.warnings import UnsupportedFieldAttributeWarning, PydanticDeprecationWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecationWarning)

import os
import dotenv
# from github.GithubException import GithubException
import asyncio
from llama_index.core.agent import AgentOutput, ToolCallResult, ToolCall
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent import FunctionAgent
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()

# Get the PR number from the environment
pr_number = os.getenv("PR_NUMBER")

# Recipe API GitHub repo url
repo_url = "https://github.com/Z-Kotliner/recipe-api.git"

# Config
# Load env variables
dotenv.load_dotenv()

# Get evn variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.environ.get('GITHUB_REPOSITORY')


# Set up logger
# logger = setup_logger("app", "DEBUG")


# Get GitHub Repository object
def get_github_repo():
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN not set")
    if not GITHUB_REPO:
        raise ValueError("GITHUB_REPO not set")

    github_client = Github(auth=Auth.Token(GITHUB_TOKEN))
    github_repo = github_client.get_repo(GITHUB_REPO)
    return github_repo


def get_llm():
    """
    LLM Provider that initializes LLM and returns a shared LLM instance.

    :return: LLM instance
    """
    llm = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_retries=2,
    )

    # llm = OpenAI("gpt-4o-mini")

    return llm


# Agents
def get_commentator_agent() -> FunctionAgent:
    # Get the tools
    add_data_to_state_tool = get_add_comment_to_state_tool()

    # Get the llm
    llm = get_llm()

    # Prompt
    prompt = """
        You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
        Ensure to do the following for a thorough review: 
            - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
            - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
            - What is good about the PR? \n
            - Did the author follow ALL contribution rules? What is missing? \n
            - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
            - Are new endpoints documented? - use the diff to determine this. \n 
            - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
            - If you need any additional details, you must hand off to the ContextAgent. \n
            - You should directly address the author. So your comments should sound like: \n      
            "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
         You must hand off to the ReviewAndPostingAgent once you are done drafting a review comment. \n
        """

    # Create ReActAgent
    commentor_agent = FunctionAgent(
        llm=llm,
        name='CommentorAgent',
        description="Uses the context gathered by the context agent to draft a pull request review comment .",
        tools=[add_data_to_state_tool],
        system_prompt=prompt,
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
    )

    return commentor_agent


def get_context_agent() -> FunctionAgent:
    # Get the tools
    pr_details_tool = get_pr_details_tool()
    commit_details_tool = get_commit_details_tool()
    file_contents_tool = get_file_contents_tool()
    add_state_tool = get_add_username_to_state_tool()

    # Get the llm
    llm = get_llm()

    # Prompt
    prompt = """
        You are the context gathering agent. When gathering context, you MUST gather \n: 
        - The details: author, title, body, diff_url, state, and head_sha; \n
        - Changed files; \n
        - Any requested for files; \n
        Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
    """

    # Create FunctionAgent
    context_agent = FunctionAgent(
        llm=llm,
        name='ContextAgent',
        description="Gathers all the needed context information",
        tools=[pr_details_tool, commit_details_tool, file_contents_tool, add_state_tool],
        system_prompt=prompt,
        can_handoff_to=["CommentorAgent"],
    )

    return context_agent


def get_review_and_post_agent() -> FunctionAgent:
    # Get the tools
    post_final_review_tool = get_post_final_review_tool()
    add_final_review_to_state_tool = get_add_final_review_to_state_tool()

    # Get the llm
    llm = get_llm()

    # Prompt
    prompt = """
        You are the Review and Posting agent. You must handoff to the CommentorAgent to create a review comment. 
        Once a review is generated, you need to run a final check and post it to GitHub.
            - The review must: \n
            - Be a ~200-300 word review in markdown format. \n
            - Specify what is good about the PR: \n
            - Did the author follow ALL contribution rules? What is missing? \n
            - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
            - Are there notes on whether new endpoints were documented? \n
            - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
         If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
         When you are satisfied, post the review to GitHub.  
           """

    # Create ReActAgent
    review_and_posting_agent = FunctionAgent(
        llm=llm,
        name='ReviewAndPostingAgent',
        description="Uses the comment by the CommentorAgent, check it, request re-write if necessary and post it to GitHub.",
        tools=[post_final_review_tool, add_final_review_to_state_tool],
        system_prompt=prompt,
        can_handoff_to=["CommentorAgent"]
    )

    return review_and_posting_agent


# Tools
async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """
    Use this tool for adding the review comment to the state.
    """
    current_state = await ctx.store.get("state")
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return f"State updated with {draft_comment} contexts. "


def get_add_comment_to_state_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=add_comment_to_state)


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """
    Use this tool for adding the final review comment to the state.
    """
    current_state = await ctx.store.get("state")
    current_state["final_review_comment"] = final_review
    await ctx.store.set("state", current_state)
    return f"State updated with {final_review} contexts. "


def get_add_final_review_to_state_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=add_final_review_to_state)


async def add_username_to_state(ctx: Context, user_name: str) -> str:
    """
    Use this tool for adding the draft comment to the state.
    """
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = user_name
    await ctx.store.set("state", current_state)
    return f"State updated with {user_name} contexts. "


def get_add_username_to_state_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=add_username_to_state)


async def get_file_contents(file_path: str):
    """Use this tool to fetch the contents of a file from a repository."""

    # Get ContentFile object
    file_content = get_github_repo().get_contents(file_path)

    return file_content.decoded_content.decode("utf-8")


def get_file_contents_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=get_file_contents)


async def post_final_review_tool(pr_number: int, comment_text: str):
    """Use this tool to post the final Pull request review comment to GitHub."""

    # Get PullRequest object
    pull_request = get_github_repo().get_pull(pr_number)

    # Find pending reviews by you
    for review in pull_request.get_reviews():
        if review.state == "PENDING":
            review.delete()

    # Post the comment on the PR
    pull_request.create_review(body=comment_text, event="COMMENT")


def get_post_final_review_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=post_final_review_tool)


async def get_pr_commit_details(commit_sha):
    """Use this tool to fetch the commit info of a given commit using commit SHA."""

    # Get Commit object
    commit = get_github_repo().get_commit(commit_sha)

    # put everything in a dict
    changed_files: list[dict[str, any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })

    return changed_files


def get_commit_details_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=get_pr_commit_details)


async def get_pr_details(pr_number: int):
    """Use this tool to fetch the pull request(PR) info with given pull request(PR) number.
      i.e Given a pull request number, get the pull request details.
    """

    # Get PullRequest object
    pull_request = get_github_repo().get_pull(pr_number)

    # put everything in a dict
    pr_details = {
        'author': pull_request.user.login,
        'title': pull_request.title,
        'body': pull_request.body,
        'diff_url': pull_request.diff_url,
        'state': pull_request.state
    }

    # Get commit SHAs
    commit_shas = []
    commits = pull_request.get_commits()

    for c in commits:
        commit_shas.append(c.sha)

    pr_details['commit_sha'] = commit_shas

    return pr_details


def get_pr_details_tool() -> FunctionTool:
    return FunctionTool.from_defaults(fn=get_pr_details)


# Agent workflow
def get_workflow_agent():
    # Get the agents
    context_agent = get_context_agent()
    commentor_agent = get_commentator_agent()
    review_and_post_agent = get_review_and_post_agent()

    # Init the workflow
    workflow_agent = AgentWorkflow(
        agents=[context_agent, commentor_agent, review_and_post_agent],
        root_agent=review_and_post_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "final_review_comment": "",
        },
    )

    return workflow_agent


# main
async def main():
    # Create question
    query = f"Write a review for PR: {pr_number}"
    prompt = RichPromptTemplate(query)

    # Get the workflow agent
    workflow_agent = get_workflow_agent()

    # Get the context
    context = Context(workflow_agent)

    # Run the agent
    handler = workflow_agent.run(prompt.format(), ctx=context)

    # Stream the output
    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
