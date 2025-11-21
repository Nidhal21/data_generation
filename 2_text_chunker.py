"""
STEP 2: Text Chunking
Loads processed documents and creates smart chunks
"""

import json
from pathlib import Path
from typing import List


class TextChunker:
    def __init__(self, output_folder: str = "output"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
    
    def smart_chunk_text(self, text: str, target_size: int = 2500) -> List[str]:
        """Smart chunking with overlap for better context"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        overlap_size = 300
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > target_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from previous chunk
                overlap = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
                current_chunk = overlap + "\n\n" + para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_chunks(self, chunk_size: int = 2500):
        """Load documents and create chunks"""
        print("=" * 80)
        print("STEP 2: TEXT CHUNKING")
        print("=" * 80)
        
        # Load processed documents
        input_file = self.output_folder / "processed_documents.json"
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            print("Run 1_document_processor.py first!")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data['documents']
        print(f"✓ Loaded {len(documents)} documents\n")
        
        # Create chunks for each document
        all_chunks = []
        chunk_id = 0
        
        for doc_idx, doc in enumerate(documents, 1):
            print(f"[{doc_idx}/{len(documents)}] {doc['filename'][:70]}")
            
            chunks = self.smart_chunk_text(doc['text'], chunk_size)
            print(f"  Created {len(chunks)} chunks")
            
            for chunk_idx, chunk_text in enumerate(chunks, 1):
                chunk_id += 1
                chunk_data = {
                    'chunk_id': chunk_id,
                    'document': doc['filename'],
                    'document_lang': doc['original_lang'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'text': chunk_text,
                    'word_count': len(chunk_text.split())
                }
                all_chunks.append(chunk_data)
        
        # Save chunks
        output_file = self.output_folder / "text_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': all_chunks,
                'stats': {
                    'total_chunks': len(all_chunks),
                    'source_documents': len(documents),
                    'chunk_size': chunk_size
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✓ Created {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    chunker = TextChunker("output")
    chunker.create_chunks(chunk_size=2500)