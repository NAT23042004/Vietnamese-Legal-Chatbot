import  dotenv
dotenv.load_dotenv()

from vector_chain import VectorChain

if __name__ == "__main__":
        #Test Vector chain
        query = """Các sản phẩm, hàng hóa nhóm 2 được miễn chứng nhận hợp quy, 
               công bố hợp quy cần đáp ứng những yêu cầu nào?"""

        vector_chain = VectorChain()
        response = vector_chain.run_vector_chain(query)
        print(response)


