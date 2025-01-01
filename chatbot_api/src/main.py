from fastapi import FastAPI
from agents.rag_agent import rag_agent_executor
from models.rag_query import QueryInput, QueryOutput
from utils.async_utils import async_retry
import  uvicorn

app = FastAPI(
    title="Vietnamese Legal Chatbot",
    description="Chatbot hỏi đáp về pháp luật Việt Nam trong lĩnh vực khoa học và công nghệ ",
)

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await rag_agent_executor.ainvoke({"input": query})

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/rag-agent")
async def query_agent(query: QueryInput) -> QueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response

if __name__ == "__main__":
    uvicorn.run(app, host= "localhost", port=8000)
