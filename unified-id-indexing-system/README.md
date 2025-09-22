# Unified ID Indexing System

This project implements a unified ID indexing system that generates and manages unique identifiers for documents and their associated metadata across various databases. The system is designed to facilitate efficient data management, indexing, and retrieval processes.

## Purpose

The primary goal of this project is to create a robust framework for generating unified IDs that can be used across different data stores, ensuring consistency and traceability of data. This enables seamless integration and interaction between various components of the system.

## Key Components

- **Core Module**: Contains classes for generating unified IDs, correlation IDs, and global sequence IDs.
  - `UnifiedID`: Manages the generation and handling of unified IDs.
  - `CorrelationID`: Generates correlation IDs for tracking requests.
  - `GlobalSequence`: Generates unique global sequence IDs.

- **Database Module**: Manages connections to various databases.
  - `DatabaseManager`: Handles interactions with MongoDB, Neo4j, Redis, and Milvus.

- **Indexing Module**: Responsible for creating and maintaining distributed indexes.
  - `DistributedIndexManager`: Manages the lifecycle of distributed indexes.
  - `IndexRegistry`: Tracks the status of indexes across different databases.

- **Document Processing Module**: Orchestrates the ingestion, preprocessing, and chunking of documents.
  - `IntegratedDocumentProcessor`: Coordinates the document processing workflow.

- **Embedding Module**: Generates embeddings for documents and nodes.
  - `EmbeddingService`: Handles the creation of embeddings.

- **Search Module**: Facilitates search operations across different data sources.
  - `UnifiedSearchEngine`: Manages search functionalities.

- **Monitoring Module**: Collects metrics and checks the health of the application.
  - `MetricsCollector`: Gathers and manages application metrics.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd unified-id-indexing-system
pip install -r requirements.txt
```

## Usage

1. **Database Setup**: Use the provided scripts to set up the required databases.
2. **Run the Application**: Start the application using Docker or directly through Python.
3. **API Access**: Access the API endpoints to interact with the indexing system.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.