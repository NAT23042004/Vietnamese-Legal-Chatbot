import os
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import  ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from  dotenv import load_dotenv

load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import numpy as np


class ChunkedEmbedding(Embeddings):
    def __init__(self, base_embedder, chunk_size=500, chunk_overlap=50):
        self.base_embedder = base_embedder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        final_embeddings = []
        for text in texts:
            # Chia văn bản thành các chunk nhỏ hơn
            chunks = self.text_splitter.split_text(text)
            if len(chunks) == 1:
                embedding = self.base_embedder.embed_documents([text])[0]
            else:
                chunk_embeddings = self.base_embedder.embed_documents(chunks)
                # Mean pooling
                embedding = np.mean(chunk_embeddings, axis=0).tolist()
            final_embeddings.append(embedding)
        return final_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.base_embedder.embed_query(text)


base_embedder = HuggingFaceEmbeddings(
    model_name="NghiemAbe/Vi-Legal-Bi-Encoder-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
chunked_embedder = ChunkedEmbedding(base_embedder)

class Neo4jVectorIndex:
    def __init__(self):
        self.base_embedder = HuggingFaceEmbeddings(
            model_name="NghiemAbe/Vi-Legal-Bi-Encoder-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.chunked_embedder = ChunkedEmbedding(self.base_embedder)
        self.vector_index = Neo4jVector.from_existing_graph(
            embedding=self.chunked_embedder,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="VectorIndex",
            node_label="Noidung",
            text_node_properties=[
                "content",
                "id",
            ],
            embedding_node_property="embedding",
        )
        self.retriever =  self.vector_index.as_retriever(k = 10)


    def get_retriever(self):
        return self.retriever

    def close_vector_index(self):
        self.vector_index._driver.close()
        return




class VectorChain:
    def __init__(self):
        self.vector_index = Neo4jVectorIndex()
        self.llm = ChatGroq(
            model=os.getenv("VECTOR_MODEL"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY")
        )
        # self.llm = GoogleGenerativeAI(
        #     model=os.getenv("GEMINI_MODEL"),
        #     api_key=os.getenv("GOOGLE_API_KEY")
        # )
        self.template = """Việc của bạn là dùng Việc của bạn là dùng các kết quả sau
            quá trình tìm kiếm bằng ngữ nghĩa (Vector Search) từ cơ sở dữ liệu Neo4j 
            để trả lời câu hỏi để trả lời câu hỏi liên quan đến pháp luật Việt Nam.
            Hãy dùng các nội dung liên quan dưới đây để trả lời câu hỏi.
            Hãy trả lời chi tiết nhất có thể và nhớ rằng đừng tự tạo ra, bịa
            ra thông tin gì hết mà không nằm trong các nội dung liên quan.
            Nếu bạn không thể trả lời câu hỏi hãy trả lời rằng bạn không biết.
            Hay nếu bạn nhận thấy rằng câu hỏi không liên quan đến pháp luật
            Việt Nam thì hãy trả lời rằng câu hỏi không nằm trong lĩnh vực mà bạn có thể trả lời.
            Hãy ghi nhớ bạn chỉ trả lời trong pháp luật Việt Nam và khi trả lời hãy trích dẫn
            các nội dung liên quan Luật nào, phần nào trong Pháp điển đến nếu tìm kiếm được bằng Vector Search vào trong câu trả lời.
            
            Chú ý quan trọng: Từ các nội dung đã trích xuất được thì 
            Nội dung liên quan: 
            {context}
            
            Câu hỏi: 
            {question}
            """

        self.prompt = ChatPromptTemplate([
            ("system", self.template),
            ("human", "{question}")
        ])

    def run_vector_chain(self, query: str) -> str:
            retriever = self.vector_index.get_retriever()
            chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
            )
            self.vector_index.close_vector_index()
            return chain.invoke(query)


if __name__ == "__main__":
    # Test Chunking Embedding class
    base_embedder = HuggingFaceEmbeddings(
        model_name="NghiemAbe/Vi-Legal-Bi-Encoder-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    chunked_embedder = ChunkedEmbedding(base_embedder)
    test_examples = ["1. Điều kiện để tàu được cấp hồ sơ đăng kiểm\
    Tàu quân sự được cơ sở đăng kiểm kiểm tra, giám sát kỹ thuật; nếu bảo đảm chất lượng an toàn kỹ thuật và bảo vệ môi trường, thỏa mãn các quy định của tiêu chuẩn quốc gia, quy chuẩn kỹ thuật quốc gia, điều ước quốc tế liên quan; quy định của Nhà nước, Bộ Quốc phòng thì được cấp hồ sơ đăng kiểm.\
    2. Hồ sơ đăng kiểm của mỗi tàu bao gồm:\
    a) Sổ kiểm tra kỹ thuật của tàu;\
    b) Giấy chứng nhận an toàn kỹ thuật và bảo vệ môi trường;\
    c) Báo cáo kiểm tra kỹ thuật của các chuyên ngành; Báo cáo kiểm tra trạng thái kỹ thuật tàu.\
    3. Thời hạn hiệu lực của hồ sơ đăng kiểm\
    a) Sổ kiểm tra kỹ thuật tàu được cấp khi kiểm tra lần đầu và sử dụng cho đến khi hết sổ. Khi thực hiện mỗi loại hình Kiểm tra, đăng kiểm viên phải xác nhận tình trạng kỹ thuật của tàu và trang thiết bị vào sổ kiểm tra kỹ thuật tàu;\
    b) Giấy chứng nhận an toàn kỹ thuật và bảo vệ môi trường được cấp cho tàu có thời hạn hiệu lực tối đa 12 (mười hai) tháng. Trước ngày hết hạn hiệu lực, tàu phải được cơ sở đăng kiểm kiểm tra đánh giá tình trạng kỹ thuật và cấp giấy chứng nhận mới;\
    c) Giấy chứng nhận an toàn kỹ thuật và bảo vệ môi trường không còn hiệu lực nêu vi phạm một trong các trường hợp sau: Tàu không được đưa vào kiểm tra theo quy định hoặc không thỏa mãn các yêu cầu kiểm tra của đăng kiểm; sau khi tàu bị thanh lý hoặc bị tai nạn, sự cố kỹ thuật không còn khả năng hoạt động; khi thay đổi kết cấu hoặc máy, trang bị kỹ thuật trong quá trình sửa chữa, hoán cải, hiện đại hóa tàu mà không có sự kiểm tra, giám sát kỹ thuật của cơ sở đăng kiểm theo quy định; vi phạm về công dụng và các điều kiện hoạt động của tàu được xác nhận trong giấy chứng nhận cấp cho tàu; các số liệu bị tẩy xóa, không rõ ràng.\
    ",
                     "Trường hợp các văn bản quy phạm pháp luật làm căn cứ, trích dẫn tại Thông tư này được sửa đổi, bổ sung hoặc thay thế bằng văn bản quy phạm pháp luật khác, thì áp dụng quy định tại văn bản ban hành mới, sửa đổi, bổ sung hoặc thay thế."]
    for test_example in test_examples:
        test_result = chunked_embedder.embed_query(test_example)
        print(test_result)

    # Test LLM from ChatGroq
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key =  os.getenv("GROQ_API_KEY")
    )
    messages = [
        (
            "system",
            "Bạn là trợ lý đắc lực trong lĩnh vực pháp luật Việt Nam",
        ),
        ("human", "Pháp luật Việt Nam có các loại văn bản nào?"),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg)

    # Test prompt and Chain
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là trợ lý đắc lực trong lĩnh vực pháp luật Việt Nam.",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    ans = chain.invoke(
        {
            "input_language": "Tiếng Việt",
            "output_language": "Tiếng Việt",
            "input": "Pháp luật Việt Nam có các loại văn bản nào?.",
        }
    )
    print(ans)