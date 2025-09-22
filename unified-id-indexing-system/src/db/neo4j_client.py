class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.connect()

    def connect(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def run_query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

    def create_node(self, label: str, properties: dict):
        query = f"CREATE (n:{label} $properties)"
        self.run_query(query, {"properties": properties})

    def create_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str):
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $start_node_id AND id(b) = $end_node_id
        CREATE (a)-[r:{relationship_type}]->(b)
        RETURN r
        """
        self.run_query(query, {"start_node_id": start_node_id, "end_node_id": end_node_id})