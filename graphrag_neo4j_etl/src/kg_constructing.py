import time
import logging
import re
from retry import retry
from typing import List, Any
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
import  os
from  dotenv import load_dotenv
from doc_processor import DocumentProcessor
import warnings
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


load_dotenv()

# Connect to Neo4j Graph Database
@retry(tries=7, delay=10)
def create_neo4j_driver(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        try:
            session.run("USE neo4j RETURN 1")
        except ClientError as e:
            if "DatabaseNotFound" in str(e):
                LOGGER.error(f"Database  not found. Please create the database manually.")
                raise e
            else:
                raise e
        driver.verify_connectivity()
    LOGGER.info("Connected to Neo4j")
    return driver

#Trích xuất số từ chuỗi dựa trên tiền tố (prefix).
def extract_number(text, prefix):
    if isinstance(text, str) and text.startswith(prefix):
        match = re.search(rf"{prefix}(\d+)$", text)
        if match:
            return int(match.group(1))  # Trích xuất và chuyển sang số nguyên
    return None

# Define Legal Knowledge Graph
class LegalKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = create_neo4j_driver(uri, user, password)

    def close(self):
        self.driver.close()

    def add_node(self, current_article, current_demuc, current_vanban):
        with self.driver.session() as session:
            # Kiểm tra và lấy số của DeMuc và Vanban
            demuc_number = None
            vanban_number = None
            demuc_exists = False
            vanban_exists = False

            if current_demuc:
                demuc_number = extract_number(current_demuc, 'demuc')
                if demuc_number is not None:
                    # Kiểm tra sự tồn tại của node DeMuc
                    query_check_demuc = (
                        "MATCH (d:DeMuc {so: $so}) "
                        "RETURN count(d) as count"
                    )
                    result = session.run(query_check_demuc, {"so": demuc_number}).single()
                    demuc_exists = result["count"] > 0
                    if not demuc_exists:
                        # Tạo node DeMuc mới nếu chưa tồn tại
                        query_create_demuc = "CREATE (d:DeMuc {so: $so})"
                        session.run(query_create_demuc, {"so": demuc_number})
                        LOGGER.info(f"Created new DeMuc node with so: {demuc_number}")
                    else:
                        LOGGER.info(f"DeMuc node with so: {demuc_number} already exists")

            if current_vanban:
                vanban_number = extract_number(current_vanban, 'vanban')
                if vanban_number is not None:
                    # Kiểm tra sự tồn tại của node Vanban
                    query_check_vanban = (
                        "MATCH (v:Vanban {so: $so, SoKyHieu: $SoKyHieu}) "
                        "RETURN count(v) as count"
                    )
                    result = session.run(query_check_vanban, {"so": vanban_number, "SoKyHieu": current_article['sohieuvanban']}).single()
                    vanban_exists = result["count"] > 0
                    if not vanban_exists:
                        # Tạo node VanBan mới nếu chưa tồn tại
                        query_create_vanban = "CREATE (v:Vanban {so: $so, SoKyHieu: $SoKyHieu})"
                        session.run(query_create_vanban, {"so": vanban_number, "SoKyHieu": current_article['sohieuvanban']})
                        LOGGER.info(f"Created new Vanban node with so: {vanban_number}, SoKyHieu: {current_article['sohieuvanban']}")
                    else:
                        LOGGER.info(f"Vanban node with so: {vanban_number}, SoKyHieu: {current_article['sohieuvanban']} already exists")

            # Tạo node NoiDung chính
            query_add_content = (
                "MERGE (n:NoiDung {"
                "sodieu: $sodieu, "
                "sohieuvanban: $sohieuvanban,"
                "tieude: $tieude, "
                "noidung: $noidung, "
                "}) "
            )
            content_params = {
                "sodieu": current_article["sodieu"],
                "sohieuvanban": current_article["sohieuvanban"],
                "tieude": current_article["tieude"],
                "noidung": current_article["noidung"],
            }
            session.run(query_add_content, content_params)

            # Tạo quan hệ DeMuc - BaoGom -> Vanban nếu cả hai node đều tồn tại
            if demuc_number is not None and vanban_number is not None:
                query_demuc_vanban = (
                    "MATCH (d:DeMuc {so: $demuc_so})"
                    "MATCH (v:Vanban {so: $vanban_so, SoKyHieu: $SoKyHieu}) "
                    "MERGE (d)-[:BaoGom]->(v)"
                )
                session.run(query_demuc_vanban, {
                    "demuc_so": demuc_number,
                    "vanban_so": vanban_number,
                     "SoKyHieu": current_article['sohieuvanban']
                })

            # Danh sách các cấp độ cần kiểm tra theo thứ tự
            hierarchy_levels = [
                ("Phan", "phan"),
                ("Chuong", "chuong"),
                ("Muc", "muc"),
                ("TieuMuc", "tieumuc"),
                ("Dieu", "dieu"),
                ("Khoan", "khoan"),
                ("Diem", "diem")
            ]

            previous_node_label = None
            previous_node_value = None
            highest_level_node = None
            highest_level_value = None

            # Kiểm tra và tạo node cho từng cấp độ
            for level_label, level_key in hierarchy_levels:
                if level_key in current_article and current_article[level_key]:
                    current_value = current_article[level_key]

                    # Kiểm tra sự tồn tại của node cấp độ
                    query_check_level = (
                        f"MATCH (n:{level_label} {{name: $name, SoKyHieu: $SoKyHieu}}) "
                        "RETURN count(n) as count"
                    )
                    result = session.run(query_check_level, {"name": current_value,  "SoKyHieu": current_article['sohieuvanban']}).single()
                    level_exists = result["count"] > 0

                    if not level_exists:
                        query_create_level = f"CREATE (n:{level_label} {{name: $name, SoKyHieu: $SoKyHieu}})"
                        session.run(query_create_level, {"name": current_value,  "SoKyHieu": current_article['sohieuvanban']})
                        LOGGER.info(f"Created new {level_label} node with so: {current_value}")
                    else:
                        LOGGER.info(f"{level_label} node with so: {current_value} already exists")

                    # Lưu node cao nhất trong phân cấp
                    if highest_level_node is None:
                        highest_level_node = level_label
                        highest_level_value = current_value

                    # Tạo quan hệ BaoGom nếu có node trước đó
                    if previous_node_label:
                        query_create_relation = (
                            f"MATCH (n1:{previous_node_label} {{name: $prev_so,  SoKyHieu: $SoKyHieu}}) "
                            f"MATCH (n2:{level_label} {{name: $curr_so,  SoKyHieu: $SoKyHieu}}) "
                            "MERGE (n1)-[:BaoGom]->(n2)"
                        )
                        session.run(query_create_relation, {
                            "prev_so": previous_node_value,
                            "curr_so": current_value,
                            "SoKyHieu": current_article['sohieuvanban']
                        })

                    previous_node_label = level_label
                    previous_node_value = current_value

            # Tạo quan hệ từ Vanban đến node cao nhất trong phân cấp hoặc trực tiếp đến NoiDung
            if vanban_number is not None:
                if highest_level_node:
                    # Tạo quan hệ từ Vanban đến node cao nhất
                    query_vanban_highest = (
                        f"MATCH (v:Vanban {{so: $vanban_so, SoKyHieu: $SoKyHieu}})"
                        f"MATCH (h:{highest_level_node} {{name: $highest_name, SoKyHieu: $SoKyHieu}}) "
                        "MERGE (v)-[:BaoGom]->(h)"
                    )
                    session.run(query_vanban_highest, {
                        "vanban_so": vanban_number,
                        "highest_name": highest_level_value,
                        "SoKyHieu": current_article['sohieuvanban']
                    })
                    LOGGER.info(f"Create a BaoGom relationship between Vanban node so: {vanban_number}, SoKyHieu: {current_article['sohieuvanban']}"
                                f" and {highest_level_node} node name: {highest_level_value}, SoKyHieu: {current_article['sohieuvanban']}")
                else:
                    # Nếu không có node phân cấp, tạo quan hệ trực tiếp đến NoiDung
                    query_vanban_content = (
                        "MATCH (v:Vanban {so: $vanban_so, SoKyHieu: $SoKyHieu}) "
                        "MATCH (n:NoiDung {sodieu: $sodieu, sohieuvanban: $SoKyHieu}) "
                        "MERGE (v)-[:BaoGom]->(n)"
                    )
                    session.run(query_vanban_content, {
                        "vanban_so": vanban_number,
                        "sodieu": current_article["sodieu"],
                        "SoKyHieu": current_article['sohieuvanban']
                    })
                    LOGGER.info(f"Create a BaoGom relationship between Vanban node so: {vanban_number}, SoKyHieu: {current_article['sohieuvanban']}"
                                f" and NoiDung node sodieu: {current_article['sodieu']}, sohieuvanban: {current_article['sohieuvanban']}")

            # Tạo quan hệ Chua từ node cuối cùng đến node NoiDung
            if previous_node_label:
                query_create_final_relation = (
                    f"MATCH (n1:{previous_node_label} {{name: $prev_name, SoKyHieu: $SoKyHieu}})"
                    f"MATCH (n2:NoiDung {{sodieu: $sodieu, sohieuvanban: $SoKyHieu}}) "
                    "MERGE (n1)-[:Chua]->(n2)"
                )
                session.run(query_create_final_relation, {
                    "prev_name": previous_node_value,
                    "sodieu": current_article["sodieu"],
                    "SoKyHieu": current_article["sohieuvanban"]
                })
                LOGGER.info(
                    f"Create a Chua relationship between {previous_node_label} node name: {previous_node_value}, SoKyHieu: {current_article['sohieuvanban']}"
                    f" and NoiDung node sodieu: {current_article['sodieu']}, sohieuvanban: {current_article['sohieuvanban']}")

    def process_folder(self, folder_path, current_demuc=None, current_vanban=None):
        folder_name = os.path.basename(folder_path)
        # Khởi tạo Document Processor
        doc_processor = DocumentProcessor()

        # Nếu thư mục là "demucX" (bắt đầu bằng "demuc"), cập nhật demuc
        if folder_name.startswith("demuc"):
            current_demuc = folder_name
            current_vanban = None  # Reset vanban khi chuyển sang demuc khác
            LOGGER.info(f"Entering demuc: {current_demuc}")

        # Nếu thư mục là "vanbanX" (bắt đầu bằng "vanban"), cập nhật vanban
        elif folder_name.startswith("vanban"):
            current_vanban = folder_name
            LOGGER.info(f"  Current vanban: {current_vanban}")

        # Duyệt qua các phần tử trong thư mục
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)

            if os.path.isdir(entry_path):
                # Nếu là thư mục, tiếp tục đệ quy
                self.process_folder(entry_path, current_demuc, current_vanban)
            elif os.path.isfile(entry_path) and entry.endswith('.docx'):
                # Nếu là file, xử lý file
                LOGGER.info(f"    Processing file: {entry} in {current_demuc}/{current_vanban}")
                doc_path = os.path.join(os.getcwd(), entry_path)
                processed_docs = doc_processor.load_document(doc_path)
                if processed_docs:
                    for processed_doc in processed_docs:
                        LOGGER.info(processed_doc)
                        # doc_embedding = create_embedding(processed_doc['noidung'], "vinai/phobert-base-v2")
                        self.add_node(processed_doc ,current_demuc, current_vanban)


def buildLegalKG():
    LOGGER.info("Building Legal Knowledge Graph...")
    knowledge_graph = LegalKnowledgeGraph(uri=os.getenv("NEO4J_URI"),
                                          user=os.getenv("NEO4J_USERNAME"),
                                          password=os.getenv("NEO4J_PASSWORD")
    )
    LOGGER.info("Adding nodes and relationships to Graph Database...")
    root_directory = "data"
    knowledge_graph.process_folder(root_directory)

    LOGGER.info("Finishing building Legal Knowledge Graph...")
    knowledge_graph.close()
    return knowledge_graph

if __name__ == '__main__':
    knowledge_graph = buildLegalKG()
