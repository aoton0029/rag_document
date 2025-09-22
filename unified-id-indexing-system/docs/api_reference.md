# API Reference for Unified ID Indexing System

## Overview
The Unified ID Indexing System provides a robust framework for generating and managing unified IDs, correlation IDs, and global sequence IDs. This document outlines the API endpoints available for interacting with the system.

## API Endpoints

### 1. Generate Unified ID
- **Endpoint**: `/api/unified-id/generate`
- **Method**: POST
- **Description**: Generates a new unified ID.
- **Request Body**:
  ```json
  {
    "source": "string" // Optional source information
  }
  ```
- **Response**:
  ```json
  {
    "unified_id": "uuid_string" // The generated unified ID
  }
  ```

### 2. Generate Correlation ID
- **Endpoint**: `/api/correlation-id/generate`
- **Method**: POST
- **Description**: Generates a new correlation ID for request tracing.
- **Request Body**:
  ```json
  {
    "request_info": "string" // Optional request information
  }
  ```
- **Response**:
  ```json
  {
    "correlation_id": "uuid_string" // The generated correlation ID
  }
  ```

### 3. Generate Global Sequence ID
- **Endpoint**: `/api/global-sequence/generate`
- **Method**: POST
- **Description**: Generates a new global sequence ID.
- **Response**:
  ```json
  {
    "global_sequence": "number" // The generated global sequence ID
  }
  ```

### 4. Get Index Status
- **Endpoint**: `/api/index/status`
- **Method**: GET
- **Description**: Retrieves the status of the indexes across different databases.
- **Response**:
  ```json
  {
    "indexes": {
      "mongodb": "status_string",
      "milvus": "status_string",
      "neo4j": "status_string",
      "redis": "status_string"
    }
  }
  ```

### 5. Health Check
- **Endpoint**: `/api/health`
- **Method**: GET
- **Description**: Checks the health status of the application and its components.
- **Response**:
  ```json
  {
    "status": "healthy" // or "unhealthy"
  }
  ```

## Error Handling
All API responses will include an error message in the following format in case of failure:
```json
{
  "error": {
    "code": "error_code",
    "message": "error_message"
  }
}
```

## Authentication
All endpoints require a valid API key to be included in the request headers:
```
Authorization: Bearer <your_api_key>
```

## Rate Limiting
The API is rate-limited to prevent abuse. Exceeding the limit will result in a `429 Too Many Requests` response.

## Conclusion
This API reference provides a comprehensive guide to the endpoints available in the Unified ID Indexing System. For further information, please refer to the deployment guide and architecture documentation.