Features
File Type Detection: Supports PDFs and image files (PNG, JPG, etc.) and applies the appropriate extraction method for each type.
Content Extraction: Uses Optical Character Recognition (OCR) for images and direct text extraction for PDFs.
Text Chunking: Splits large text into manageable chunks with overlap to maintain context.
Embedding Generation: Converts text chunks into vector embeddings using OpenAI embeddings for similarity searches.
Vector Indexing: Stores embeddings in FAISS for efficient similarity-based retrieval.
Question Answering: Processes user queries to retrieve relevant content and generate concise answers with confidence scores.
