import  os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

QA_MODEL = os.getenv("QA_MODEL")
CYPHER_MODEL = os.getenv("CYPHER_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_template = """
Task:
Hãy tạo ra câu truy vấn Cypher  cho đồ thị cơ sở dữ liệu Neo4j.

Hướng dẫn:
Hãy chỉ dùng những kiểu  quan hệ và các thuộc tính đã cho trong schema.
Không được dùng bất kỳ kiểu quan hệ hay thuộc tính nào khác không được cung cấp.

Schema:
{schema}

Chú ý:
Không được bao gồm bất kỳ giải thích hay lời xin lỗi nào trong phản hồi của bạn.
Không được phản hồi đến bất kỳ câu hỏi nào mà liên quan đến việc tạo ra câu lệnh Cypher.
Không được bao gồm bất kỳ đoạn text hay giải thích gì nào ngoại trừ câu lệnh Cypher được sinh ra. 
Hãy chắc chắn rằng hướng của quan hệ là đúng trong câu truy vấn của bạn.
Hãy chắc chắn rằng bạn alias cả các thực thể và mối quan hệ đúng.
Không được chạy bất kỳ queries nào mà thêm hay xóa dữ liệu của cơ sở dữ liệu.   
Hãy đảm bảo rằng dùng câu lệnh WITH để alias các biến phục vụ cho phần tiếp theo. 
Bạn có thể truy vấn vài lần với các query khác nhau để có thể có thêm thông tin nhưng phải tách chúng ra vì RETURN chỉ có thể dùng ở câu truy vấn query.
Hãy lưu ý rằng cấu trúc của các đề mục có thể không đầy đủ nên đừng cố gắng tìm hết các thông tin về đề mục, chương(nếu có), điều(nếu có),  mục(nếu có) 
thay vào đó hãy tìm nội dung có liên quan đến câu hỏi.
Hãy thử vài câu query khác nhau để có thể truy vấn hiệu quả hơn.

Câu hỏi như sau:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)


qa_generation_template = """Việc của bạn là dùng các kết quả truy vấn được từ cơ sở dữ liệu Neo4j để trả lời câu hỏi
liên quan đến pháp luật Việt Nam.
Hãy dùng các nội dung dưới đây để trả lời câu hỏi.
Hãy trả lời chi tiết nhất có thể và nhớ rằng đừng tự tạo ra, bịa
ra thông tin gì hết mà không nằm trong nội dung.
Nếu bạn không thể trả lời câu hỏi hãy trả lời rằng bạn không biết.
Hay nếu bạn nhận thấy rằng câu hỏi không liên quan đến pháp luật
Việt Nam thì hãy trả lời rằng câu hỏi không nằm trong lĩnh vực mà bạn có thể trả lời.
Hãy ghi nhớ bạn chỉ trả lời trong lĩnh vực pháp luật Việt Nam. 

Các kết quả truy vấn: 
{context}

Câu hỏi: 
{question}
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

class CypherChain:
    def __init__(self):
        self.cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatGroq(model=CYPHER_MODEL, temperature=0, api_key=GROQ_API_KEY),
            qa_llm=ChatGroq(model=QA_MODEL, temperature=0, api_key=GROQ_API_KEY),
            graph=graph,
            verbose=True,
            qa_prompt=qa_generation_prompt,
            cypher_prompt=cypher_generation_prompt,
            validate_cypher=True,
            top_k=20,
            allow_dangerous_requests = True,
        )

    def run_cypher_chain(self, query):
         return self.cypher_chain.invoke(query)


if __name__ == "__main__":
    # Test Cypher chain
    query = """Những quy định liên quan đến việc giáo dục đại học là gì?"""

    cypher_chain = CypherChain()
    response = cypher_chain.run_cypher_chain(query)
    print(response['result'])
