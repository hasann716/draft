# BIM Magic
The system interacts with BIM (Building Information Modeling) data, and provides answer to questions related to the building project.   
The BIM data ( IFC format), is pre loaded to Neo4j DB, with its entities and relashionship.   
The system learns the relashionships from prompt question and answer, and respond in human language.  
try it on: https://bim-robot.streamlit.app/ (partial small deployed on [Neo4j cloud](https://neo4j.com/))   

![image](https://github.com/user-attachments/assets/af44c4b8-41dc-40e4-b3f4-e3580b1ff243)

# Hige Level Architecture   
![image](https://github.com/user-attachments/assets/33266411-8136-4591-a44f-d6a2ae057fda)


# Requirements
- [Poetry](https://python-poetry.org) for dependency managament.
- Load IFC file to Neo4j using IFCSell https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifctopologyresource/lexical/ifcshell.htm   
- Duplicate the `secrets.toml.example` file to `secrets.toml` and populate with appropriate keys.
see: https://bim-robot.streamlit.app/

## Usage
poetry update
poetry run streamlit run bim_streamlit/main.py --server.port=80
# draft
# draft
