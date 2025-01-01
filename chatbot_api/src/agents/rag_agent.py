import os
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from chains.vector_chain import VectorChain
from chains.cypher_chain import CypherChain
from langchain_core.prompts import PromptTemplate


AGENT_MODEL = os.getenv("AGENT_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


tools = [
    Tool(
        name="Vector Search",
        func=VectorChain.run_vector_chain,
        description="""Công cụ hữu ích giúp bạn trả lời các câu hỏi
        liên quan đến hỏi đáp về Pháp luật Việt Nam trong lĩnh vực khoa học
        và công nghệ bằng cách tìm kiếm bằng ngữ nghĩa (semantic search).
        Hãy sử dụng toàn bộ cái prompt này như là đầu vào cho công cụ.
        Ví dụ nếu promt là "Các nguyên tắc quản lý sản phẩm hàng hóa nhóm 2 là gì?",
        thì câu hỏi đầu vào sẽ là "Các nguyên tắc quản lý sản phẩm hàng hóa nhóm 2 là gì?".
        """,
    ),
    Tool(
        name="Cypher Chain",
        func=CypherChain.run_cypher_chain,
        description="""Công cụ hữu ích giúp bạn trả lời các câu hỏi
        liên quan đến hỏi đáp về Pháp Luật Việt Nam trong lĩnh vực khoa học
        và công nghệ dựa trên việc tìm kiếm trên Cypher của Neo4j Graph Database.
        Hãy sử dụng toàn bộ cái prompt này như là đầu vào cho công cụ.
        Ví dụ nếu promt là "Các nguyên tắc quản lý sản phẩm hàng hóa nhóm 2 là gì?",
        thì câu hỏi đầu vào sẽ là "Các nguyên tắc quản lý sản phẩm hàng hóa nhóm 2 là gì?".
        """,
    ),
]


chat_model = ChatGroq(model=AGENT_MODEL,temperature=0, api_key=GROQ_API_KEY)
# chat_model_with_tools = chat_model.bind_tools(tools)
# response = chat_model_with_tools.invoke([HumanMessage(content="Pháp luật Việt Nam có các loại văn bản nào?.")])
# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

agent_prompt_template = """
Hãy dùng kết hợp hai công cụ Vector Search và Cypher Chain để phục vụ cho 
việc trả lời các câu hỏi liên quan đến hỏi đáp về Pháp Luật Việt Nam trong lĩnh vực khoa học
và công nghệ. 
Hãy dựa vào kết quả từ hai công cụ hãy trả lời câu hỏi. 
Hãy trả lời chi tiết nhất có thể và nhớ rằng đừng tự tạo ra, bịa
ra thông tin gì hết mà không nằm trong nội dung.
Nếu bạn không thể trả lời câu hỏi hãy trả lời rằng bạn không biết.
Hay nếu bạn nhận thấy rằng câu hỏi không liên quan đến pháp luật
Việt Nam trong lĩnh vực khoa học và công nghệ thì hãy trả lời rằng
câu hỏi không nằm trong lĩnh vực mà bạn có thể trả lời.

{agent_scratchpad}
"""
agent_prompt = PromptTemplate(template=agent_prompt_template)

rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=agent_prompt,
    tools=tools,
)

rag_agent_executor = AgentExecutor(
    agent=rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)

if __name__ == "__main__":
    input_data = {"messages": [HumanMessage(
        content="Các nguyên tắc quản lý sản phẩm, hàng hóa nhóm 2 là gì?")]}
    config = {"recursion_limit": 25}
    response = rag_agent_executor.invoke(input_data, config)
    response.get("output")

    response = rag_agent_executor.invoke(
        {
            "input": (
                "Các nguyên tắc quản lý sản phẩm, hàng hóa nhóm 2 là gì?"
            )
        }
    )
