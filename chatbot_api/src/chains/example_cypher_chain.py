from cypher_chain import CypherChain

if __name__ == "__main__":
    # Test Cypher chain
    query = """Các sản phẩm, hàng hóa nhóm 2 được miễn chứng nhận hợp quy, 
                   công bố hợp quy cần đáp ứng những yêu cầu nào?"""

    cypher_chain = CypherChain()
    response = cypher_chain.run_cypher_chain(query)
    print(response)
