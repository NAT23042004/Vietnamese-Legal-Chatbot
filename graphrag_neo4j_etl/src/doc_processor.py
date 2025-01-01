from langchain_community.document_loaders import Docx2txtLoader
from typing import List, Dict, Optional
from langchain_core.documents import Document

import os
import re

regex_sohieu_patterns = 'Số:\s*\d+/\d+/[A-ZĂÂĐÊÔƠƯ]+-[A-ZĂÂĐÊÔƠƯ]+-[A-ZĂÂĐÊÔƠƯ]+|Số:\s*\d+/\d+/[A-ZĂÂĐÊÔƠƯ]+-[A-ZĂÂĐÊÔƠƯ]+|Số:\s*\d+/\d+/[A-ZĂÂĐÊÔƠƯ]+'
class DocumentProcessor:
    """Process DOCX and PDF files and extract structured content using LangChain and OCR."""

    def __init__(self):
        self.vietnamese_capital_letters = "ĂÂĐÊÔƠƯ"
        self.pattern1 = fr"""Số:\s*\d+\s*/\d+/[A-Z{self.vietnamese_capital_letters}]+-[A-Z{self.vietnamese_capital_letters}]+-[A-Z{self.vietnamese_capital_letters}]+"""
        self.pattern2 = fr"""Số:\s*\d+\s*/\d+/[A-Z{self.vietnamese_capital_letters}]+-[A-Z{self.vietnamese_capital_letters}]+"""
        self.pattern3 = fr"""Số:\s*\d+\s*/\d+/[A-Z{self.vietnamese_capital_letters}]+"""
        self.pattern = fr"{self.pattern1}|{self.pattern2}|{self.pattern3}"
        self.document_number = None
        self.signature_patterns = [
            r"Nơi nhận:", r"TM.", r"KT.", r"CHỦ TỊCH", r"BỘ TRƯỞNG", r"GIÁM ĐỐC",
            r"THỦ TRƯỞNG", r"TỔNG GIÁM ĐỐC", r"TRƯỞNG BAN", r"CHÁNH VĂN PHÒNG",
            r"XÁC NHẬN", r"PHÊ DUYỆT", r"QUYẾT ĐỊNH", r"THÔNG QUA", r"ĐỒNG Ý"
        ]
        self.signature_regex = re.compile('|'.join(self.signature_patterns))

    def _is_signature_section(self, text: str) -> bool:
        return bool(self.signature_regex.search(text))

    def _extract_document_number(self, text: str) -> Optional[str]:
        if "Số:" in text:
            matches = re.findall(self.pattern, text)
            if matches:
                return matches[0][3:]
        return None

    def load_document(self, file_path: str) -> Optional[List[Dict]]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
                document = loader.load()[0]
                return self._process_content(document)
            else:
                print(f"Unsupported file format: {file_extension}")
                return None

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return None


    def split_article_content(self, article: Dict) -> List[Dict]:
        """
        Chia nội dung của điều luật thành các đoạn nhỏ dựa trên số thứ tự

        Parameters:
        article: Dict - Điều luật cần xử lý

        Returns:
        List[Dict] - Danh sách các đoạn văn đã được tách
        """
        # Kiểm tra nếu nội dung rỗng
        if not article["noidung"]:
            return [article]

        # Pattern để tìm các đoạn bắt đầu bằng số
        pattern = r'(?:^|\s)(\d+\.[\s\S]*?)(?=\s\d+\.|$)'

        # Tìm tất cả các đoạn trong nội dung
        matches = re.finditer(pattern, article["noidung"])
        paragraphs = [match.group(1).strip() for match in matches]

        # Nếu không tìm thấy đoạn nào phù hợp với pattern, trả về article gốc
        if not paragraphs:
            return [article]

        # Tạo danh sách các article mới, mỗi article chứa một đoạn
        split_articles = []
        for idx, paragraph in enumerate(paragraphs, 1):
            new_article = article.copy()
            new_article["noidung"] = paragraph
            new_article["sodieu"] = f"{article['sodieu']}.{idx}"  # Thêm số thứ tự phụ
            split_articles.append(new_article)

        return split_articles

    def _process_content(self, document) -> List[Dict]:
        data = []
        current_article = None
        current_section = None
        current_chapter = None
        current_part = None
        current_subsection = None
        found_first_article = False
        last_article_number = None

        lines = document.page_content.split('\n')

        for line in lines:
            text = line.strip()
            if not text:
                continue

            if self._is_signature_section(text):
                if current_article:
                    # Chia nhỏ nội dung trước khi thêm vào data
                    split_articles = self.split_article_content(current_article)
                    data.extend(split_articles)
                break

            if not found_first_article:
                doc_num = self._extract_document_number(text)
                if doc_num:
                    # Removing Leading and Trailing Spaces and then remove the remaining spaces
                    self.document_number = doc_num.strip().replace(" ", "")
                    print(f"Found document number: {doc_num}")

            if text.startswith("Phần"):
                current_part = text
                print(f"Found new part: {current_part}")
                continue

            if text.startswith("Chương"):
                current_chapter = text
                print(f"Found new chapter: {current_chapter}")
                continue

            if text.startswith("Mục"):
                current_section = text
                print(f"Found new section: {current_section}")
                continue

            if text.startswith("Tiểu mục"):
                current_subsection = text
                print(f"Found new subsection: {current_subsection}")
                continue

            if text.startswith("Điều"):
                found_first_article = True

                if current_article:
                    # Chia nhỏ nội dung trước khi thêm vào data
                    split_articles = self.split_article_content(current_article)
                    data.extend(split_articles)

                parts = text.split(" ", 1)
                extracted_part = parts[1].split(".", 1)
                article_number = extracted_part[0]
                title = extracted_part[1] if len(extracted_part) > 1 else ""

                current_article = {
                    "sohieuvanban": self.document_number,
                    "sodieu": article_number,
                    "tieude": title,
                    "noidung": "",
                    "phan": current_part,
                    "chuong": current_chapter,
                    "muc": current_section,
                    "tieumuc": current_subsection
                }
                last_article_number = article_number
            elif current_article is not None:
                current_article["noidung"] += text + " "

        if current_article and not self._is_signature_section(text):
            # Chia nhỏ nội dung cuối cùng trước khi thêm vào data
            split_articles = self.split_article_content(current_article)
            data.extend(split_articles)

        return data

if __name__ == "__main__":
    processor = DocumentProcessor()
    doc_path = os.path.join(os.getcwd(), 'data',  'chude19', 'demuc1', 'vanban30', '2-DTThongtu(G)-TTNSNTky.docx')
    print(os.getcwd())
    processed_doc = processor.load_document(doc_path)
    if processed_doc:
        for entry in processed_doc:
            print(entry)
