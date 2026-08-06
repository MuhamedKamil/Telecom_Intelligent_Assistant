from pathlib import Path
from datetime import datetime
import mimetypes

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from typing import Optional, Dict, List, Any, Union


class DocumentFormatter:
    """
    Formats document chunks into a structured output with metadata.
    Takes raw chunks from a file and organizes them with position tracking,
    text statistics, and file information.
    """

    def format(self, file_path: str, chunks: list) -> dict:
        """
        Process and structure document chunks.
        
        Args:
            file_path (str): Path to the source document
            chunks (list): List of document chunk objects with text and metadata
        
        Returns:
            dict: Contains:
                - document: File info (name, type, total chunks)
                - chunks: List of formatted chunks with text, language, and stats
                - processing_metadata: Processing details (strategy, timestamp)
        """
        path = Path(file_path)

        mime_type, _ = mimetypes.guess_type(path)

        formatted_chunks  = []
        current_position  = 0
        total_chunks      = len(chunks)

        for idx, chunk in enumerate(chunks):

            text  = chunk.text.strip()
            start = current_position
            end   = start + len(text)

            formatted_chunks.append({
                "chunk_id"     : f"chunk_{idx + 1:03d}",
                "chunk_index"  : idx,
                "total_chunks" : total_chunks,
                "text"         : text,
                "lang"         : chunk.metadata.languages[0] if chunk.metadata.languages else "unknown",
                "metadata": {
                    "page_number" : getattr(chunk.metadata, "page_number", None),
                    "token_count" : len(text.split()),
                    "word_count"  : len(text.split()),
                    "char_count"  : len(text),
                    "sentence_count": (
                        text.count(".")
                        + text.count("!")
                        + text.count("?")
                    ),
                    "start_position" : start,
                    "end_position"   : end,
                },
            })

            current_position = end

        return {
            "document": {
                "source"       : "uploaded",
                "file_name"    : path.name,
                "file_type"    : mime_type,
                "last_modified": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat(),
                "total_chunks": total_chunks,
            },
            "chunks": formatted_chunks,
            "processing_metadata": {
                "chunking_strategy": "unstructured_chunk_by_title",
                "processed_at"     : datetime.utcnow().isoformat(),
            },
        }


class DocumentPipeline:
    """
    A pipeline for processing documents: extracts text from files, splits it into chunks,
    and formats the output with metadata.
    
    Supports PDF, DOCX, TXT, PNG, JPG, and JPEG files.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
  
    }

    def __init__(
        self,
        document_config: Optional[Dict] 

    ):
        """
        Initialize the pipeline with chunking parameters.
        
        Args:
            max_characters (int): Maximum characters per chunk. Default: 1000
            new_after_n_chars (int): Start new chunk after this many chars. Default: 800
            overlap (int): Character overlap between chunks. Default: 100
        """

        self.max_characters      = document_config["max_characters"]
        self.new_after_n_chars   = document_config["new_after_n_chars"]
        self.overlap             = document_config["overlap"]

        self.formatter = DocumentFormatter()

    def process(self, file_path: str) -> dict:
        """
        Process a document file and return formatted chunks.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            dict: Formatted document data with chunks and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}"
            )

        elements = partition(filename=str(path))

        chunks = chunk_by_title(
            elements,
            max_characters    = self.max_characters,
            new_after_n_chars = self.new_after_n_chars,
            overlap           = self.overlap,
        )

        return self.formatter.format(str(path), chunks)



sample_file = "bills.txt"
# Now process it
pipeline = DocumentPipeline({"max_characters": 200, "new_after_n_chars": 150, "overlap": 50})
result = pipeline.process(sample_file)

# Print the first chunk's text
print(f"First chunk: {result['chunks'][0]['text']}")
print(f"Total chunks: {result['document']['total_chunks']}")