from langchain.chains import GraphCypherQAChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain_google_genai import ChatGoogleGenerativeAI
from retry import retry
import logging
import streamlit as st
from common_functions import ChainClass

#CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database strictly based on the schema and instructions provided.
#Instructions:
#1. Use only nodes, relationships, and properties mentioned in the schema.
#2. Always enclose the Cypher output inside 3 backticks. Do not add 'cypher' after the backticks.
#3. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Company name use `toLower(c.name) contains 'neo4j'`
#4. Always use aliases to refer the node in the query
#5. Always return count(DISTINCT n) for aggregations to avoid duplicates
#6. `OWNS_STOCK_IN` relationship is syonymous with `OWNS` and `OWNER`
#7. Use examples of questions and accurate Cypher statements below to guide you.
#Schema:
#{schema}

CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only Node properties defined under the schema.
use only cypher relationships that are mentioned below "The relationships:" 
All the data is based on ifc (industry foundation classes) standard.
Do not use any other relationship types or properties that are not provided.
use the examples below as much as possible, before submitting the answer.  
make sure that you are writing correct cypher syntax, and dont mix with SQL syntax!!!!!


Cypher examples:
examples:
# how many floors in the building?
response: MATCH (b:IfcBuilding)<-[:RelatingObject]-(:IfcRelAggregates)-[:RelatedObjects]->(bs:IfcBuildingStorey) RETURN count(DISTINCT bs)
# get gross volume of all entities that has gross volume (such as IfcBeam, IfcWallStandardCase, IfcSlab, IfcColumn): 
resonses:            MATCH (n:IfcQuantityVolume)<-[:Quantities]-(:IfcElementQuantity)<-[:RelatingPropertyDefinition]-(:IfcRelDefinesByProperties)-[:RelatedObjects]->(o) return labels(o)[0] as entity_name,o.ObjectType as entity_type ,n.VolumeValue as gross_volume
# get all columns with relation to floors:
resonses:            MATCH (n:IfcBuildingStorey)<-[:RelatingStructure]-(ss:IfcRelContainedInSpatialStructure)-[:RelatedElements]->(c:IfcColumn) RETURN n,ss,c
# whar are the materials used for beam?
resonses:             match (mt:IfcMaterial)<-[:RelatingMaterial]-(ascmat:IfcRelAssociatesMaterial)-[:RelatedObjects]->(bm:IfcBeam ) return (mt) .
# what objects were made by materials ?
resonses:            MATCH (m:IfcMaterial)<-[:RelatingMaterial]-(asm:IfcRelAssociatesMaterial)-[:RelatedObjects]->(o) return (o)
# what spaces spaces are in floors?
resonses:            match(IfcBuildingStorey)<-[:RelatingObject]-(:IfcRelAggregates)-[:RelatedObjects]->(s:IfcSpace) return (s)
# count beams by type:
resonses:           MATCH (b:IfcBeam) RETURN b.ObjectType AS object_type, COUNT(b) AS count ORDER BY count DESC
# count columns by type:
resonses:            MATCH (b:IfcColumns) RETURN b.ObjectType AS object_type, COUNT(b) AS count ORDER BY count DESC
# count walls by type:
resonses:            MATCH (b:IfcWall) RETURN b.ObjectType AS object_type, COUNT(b) AS count ORDER BY count DESC
# get the buildings floor/storey sequence, Storey Level, Storey Elevation and Storey Name:
resonses:            MATCH (storey:IfcBuildingStorey) WHERE storey.Elevation IS NOT NULL WITH storey ORDER BY storey.Elevation ASC WITH collect(storey) AS sortedStoreys UNWIND range(0, size(sortedStoreys) - 1) AS idx WITH idx, sortedStoreys[idx] AS storey, sortedStoreys RETURN CASE WHEN storey.Elevation < 0 THEN -1 * (size([s IN sortedStoreys WHERE s.Elevation < 0]) - idx) WHEN storey.Elevation = 0 THEN 0 ELSE idx - size([s IN sortedStoreys WHERE s.Elevation < 0]) END AS StoreyLevel, idx+1 as StoreySequence,storey.Name AS StoreyName, storey.Elevation AS Elevation ORDER BY storey.Elevation ASC
# what are the relation between IfcBuilding and  IfcBuildingStorey storeys:
resonses:            MATCH (b:IfcBuilding)<-[:RelatingObject]-(:IfcRelAggregates)-[:RelatedObjects]->(bs:IfcBuildingStorey) return(bs)
# get the building top floor/storey name:
resonses:            MATCH (storey:IfcBuildingStorey) WHERE storey.Elevation IS NOT NULL WITH storey ORDER BY storey.Elevation DESC LIMIT 1 RETURN storey.Name AS TopStoreyName
# get the  storey level of the middle storey:
resonses:            MATCH (storey:IfcBuildingStorey) WHERE storey.Elevation IS NOT NULL WITH storey ORDER BY storey.Elevation ASC WITH collect(storey) AS sortedStoreys WITH sortedStoreys, size(sortedStoreys) AS totalCount WITH sortedStoreys, totalCount, floor(totalCount / 2.0) AS middleIndex RETURN sortedStoreys[toInteger(middleIndex)] AS middleStorey, sortedStoreys[toInteger(middleIndex)].Name AS StoreyName, sortedStoreys[toInteger(middleIndex)].Elevation AS Elevation
# get the number of columns, sum of column volume, sum of beam volume,  average of beam volume,  average of column volume for each storey of the building, mention the storey name and elevation order by elevation:
resonses:            MATCH (storey:IfcBuildingStorey)<-[:RelatingStructure]-(ss:IfcRelContainedInSpatialStructure)-[:RelatedElements]->(c:IfcColumn) OPTIONAL MATCH (c)<-[:RelatedObjects]-(eq:IfcRelDefinesByProperties)-[:RelatingPropertyDefinition]->(ev:IfcElementQuantity)-[:Quantities]->(qv:IfcQuantityVolume) WITH storey, count(c) AS column_count, sum(qv.VolumeValue) AS column_volume_sum, storey.Name AS storey_name, storey.Elevation AS storey_elevation OPTIONAL MATCH (storey)<-[:RelatingStructure]-()-[:RelatedElements]->(b:IfcBeam) OPTIONAL MATCH (b)<-[:RelatedObjects]-()-[:RelatingPropertyDefinition]->(ev:IfcElementQuantity)-[:Quantities]->(qv:IfcQuantityVolume) WITH storey, column_count, column_volume_sum, storey_name, storey_elevation, count(b) AS beam_count, sum(qv.VolumeValue) AS beam_volume_sum RETURN storey_name, storey_elevation, column_count, column_volume_sum, beam_count, beam_volume_sum, CASE WHEN beam_count > 0 THEN beam_volume_sum / beam_count ELSE 0 END AS beam_volume_avg, CASE WHEN column_count > 0 THEN column_volume_sum / column_count ELSE 0 END AS column_volume_avg ORDER BY storey_elevation ASC
$ get materials of slab type
resonses:            MATCH (m)<-[:Material]-(ml:IfcMaterialLayer)<-[:MaterialLayers]-(:IfcMaterialLayerSet)<-[:RelatingMaterial]-(:IfcRelAssociatesMaterial)-[:RelatedObjects]->(st:IfcSlabType) return m,st
# get all elements that use steel
resonses:            MATCH (m:IfcMaterial)<-[:RelatingMaterial]-(ascmat:IfcRelAssociatesMaterial)-[:RelatedObjects]->(o) WHERE toLower(m.Name) CONTAINS 'steel' RETURN o
# get element , element type, element entity and its related implicit quantity names (example: GrossVolume, length, area) and their coresponding quantity values
resonses:            MATCH (e)<-[:RelatedObjects]-(rdbp:IfcRelDefinesByProperties)-[:RelatingPropertyDefinition]->(eq:IfcElementQuantity)-[:Quantities]->(q) WITH e, eq, q, LABELS(e)[0] AS element_entity, PROPERTIES(q) AS qProps RETURN e.Name AS elementName, element_entity ,e.ObjectType as element_type ,q.Name AS quantity_property_name, CASE WHEN size(keys(qProps)) >= 2 THEN qProps[ keys(qProps)[1] ] ELSE null END AS quantity_property_value
# what is the height of קומה 15?
response:           MATCH (n:IfcBuildingStorey {{Name: "קומה 15"}}) return n.Elevation      
# slab and slab thinkness for all slabs greater than 25 
response:           match (s:IfcSlab)<-[:RelatedObjects]-(:IfcRelAssociatesMaterial)-[:RelatingMaterial]->(:IfcMaterialLayerSetUsage)-[:ForLayerSet]->(:IfcMaterialLayerSet)-[:MaterialLayers]->(ml:IfcMaterialLayer) where ml.LayerThickness> 25.0 return s, ml.LayerThickness 
# all slabs located on floor named קומה 9   
response:           MATCH (storey:IfcBuildingStorey {{Name: "קומה 9"}} )<-[:RelatingStructure]-(ss:IfcRelContainedInSpatialStructure)-[:RelatedElements]->(c:IfcSlab) return c
# get list of slabs and  the slab area
response:           MATCH (c:IfcSlab)<-[:RelatedObjects]-(eq:IfcRelDefinesByProperties)-[:RelatingPropertyDefinition]->(ev:IfcElementQuantity)-[:Quantities]->(qv:IfcQuantityArea {{Name: "NetArea"}}) return c, qv.AreaValue
# get list of slabs and slab types: 
response:           MATCH (s:IfcSlab)<-[:RelatedObjects]-(:IfcRelDefinesByType)-[:RelatingType]->(st:IfcSlabType) return s, st   

Critical Note: 
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

MEMORY = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key='query', 
    output_key='result', 
    return_messages=True,
    max_token_limit=20000)

url = st.secrets["NEO4J_URI"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]

graph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
    sanitize = True
)
import os

class CypherChainClass(ChainClass):
    def set_chain(self):
        print("setting new graphchain")
        print(self.model_name, self.api_base, self.api_key)
        self.graph_chain=None
        if "gemini" in self.model_name:
            cypher_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key,temperature=0, verbose=True,top_k=200)
            qa_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key,temperature=0, verbose=True,top_k=200)
        else:
            cypher_llm = ChatOpenAI(model=self.model_name, openai_api_key=self.api_key,openai_api_base=self.api_base,temperature=0)
            qa_llm= ChatOpenAI(model=self.model_name, openai_api_key=self.api_key,openai_api_base=self.api_base,temperature=0)
        self.graph_chain = GraphCypherQAChain.from_llm(
            cypher_llm=cypher_llm,
            top_k=200,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            qa_llm=qa_llm,
            validate_cypher= True,
            graph=graph,
            memory=MEMORY,
            verbose=True, 
            return_intermediate_steps = True,
            return_direct = True)


    @retry(tries=1, delay=12)
    def get_results(self, question) -> str:
        """Generate a response from a GraphCypherQAChain targeted at generating answered related to relationships. 

        Args:
            question (str): User query

        Returns:
            str: Answer from chain
        """

        logging.info(f'Using Neo4j database at url: {url}')

        graph.refresh_schema()
        
        prompt=CYPHER_GENERATION_PROMPT.format(schema=graph.get_schema, question=question)
        #print('Prompt:', prompt)

        chain_result = None
#        print("before chain result")
        try:
            with get_openai_callback() as cb:
                chain_result = self.graph_chain.invoke({
                    "query": question},
                    prompt=prompt,
                    return_only_outputs = True,
                )
                print(cb)
            logging.info(f'chain result: {chain_result}')
            print(f'chain result: {chain_result}')
        except Exception as e:
            # Occurs when the chain can not generate a cypher statement
            # for the question with the given database schema
            logging.warning(f'Handled exception running graphCypher chain: {e}')
            print(e)
        logging.info(f'chain_result: {chain_result}')

        if chain_result is None:
            return "Sorry, I couldn't find an answer to your question"
        
        result = chain_result.get("result", None)
        intermediate_steps=chain_result.get("intermediate_steps", None)

        return result, intermediate_steps, cb
