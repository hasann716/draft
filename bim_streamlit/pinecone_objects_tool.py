from langchain.tools import tool
from pinecone_objects_chain import RagChainClass

class PineconeObjectsTool:
    def __init__(self):
        self.cypher_chain = RagChainClass()

#    @tool("graph-cypher-tool") - AMIT when working with agent this would need to be changed!!
    def run(self, tool_input:str) -> str:
        """
        For finding similar entities to the ones in the search query.
        """
#        print("Tool input is:" + tool_input)
        return (self.cypher_chain.get_results(tool_input))