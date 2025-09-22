class DocumentChunker:
    """
    DocumentChunker class handles the chunking of documents into smaller pieces
    based on specified parameters such as chunk size and overlap.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document_text: str) -> List[str]:
        """
        Splits the document text into chunks based on the specified chunk size and overlap.

        Args:
            document_text: The text of the document to be chunked.

        Returns:
            A list of text chunks.
        """
        chunks = []
        start = 0
        while start < len(document_text):
            end = start + self.chunk_size
            chunk = document_text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap  # Move start index for next chunk
        return chunks

    def chunk_documents(self, documents: List[str]) -> Dict[str, List[str]]:
        """
        Chunks multiple documents.

        Args:
            documents: A list of document texts to be chunked.

        Returns:
            A dictionary where keys are document identifiers and values are lists of chunks.
        """
        chunked_documents = {}
        for doc_id, doc_text in enumerate(documents):
            chunked_documents[f'doc_{doc_id}'] = self.chunk_document(doc_text)
        return chunked_documents