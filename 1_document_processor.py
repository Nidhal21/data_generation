"""
STEP 1: Document Extraction and Translation
Processes documents and saves extracted text to JSON
"""

import os
import json
import re
from pathlib import Path
from typing import Tuple
import time

try:
    from PyPDF2 import PdfReader
    from docx import Document
    from deep_translator import GoogleTranslator
except ImportError:
    print("Installing required packages...")
    os.system("pip install PyPDF2 python-docx deep-translator")
    from PyPDF2 import PdfReader
    from docx import Document
    from deep_translator import GoogleTranslator


class DocumentProcessor:
    def __init__(self, docs_folder: str = "Docs", output_folder: str = "output"):
        self.docs_folder = Path(docs_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        self.language_stats = {"english": 0, "arabic": 0, "french": 0}
        self.translation_stats = {"arabic_translated": 0, "french_translated": 0}
        
        # Initialize translators
        self.ar_translator = GoogleTranslator(source='ar', target='en')
        self.fr_translator = GoogleTranslator(source='fr', target='en')
    
    def detect_language(self, text: str) -> str:
        """Detect language with improved accuracy"""
        sample = text[:800].lower()
        
        # Arabic detection
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', sample))
        if arabic_chars > 20:
            return 'arabic'
        
        # French detection
        french_words = ['le', 'la', 'les', 'de', 'et', 'dans', 'pour', 'est', 'que', 'une', 'des', 'sont']
        french_count = sum(1 for w in french_words if f' {w} ' in sample)
        
        # English detection
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'are', 'that', 'for']
        english_count = sum(1 for w in english_words if f' {w} ' in sample)
        
        if french_count > english_count + 2:
            return 'french'
        
        return 'english'
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English in chunks"""
        if source_lang == 'english':
            return text
        
        # Split into paragraphs for better translation
        paragraphs = text.split('\n\n')
        translated_paras = []
        
        translator = self.ar_translator if source_lang == 'arabic' else self.fr_translator
        
        for para in paragraphs:
            if len(para.strip()) < 20:
                continue
            
            try:
                # Translate in chunks (max 4500 chars per request)
                if len(para) > 4500:
                    chunks = [para[i:i+4500] for i in range(0, len(para), 4500)]
                    translated = ' '.join([translator.translate(chunk) for chunk in chunks])
                else:
                    translated = translator.translate(para)
                
                translated_paras.append(translated)
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"  ⚠ Translation error: {e}")
                continue
        
        if source_lang == 'arabic':
            self.translation_stats['arabic_translated'] += 1
        else:
            self.translation_stats['french_translated'] += 1
        
        return '\n\n'.join(translated_paras)
    
    def extract_text_from_file(self, filepath: Path) -> Tuple[str, str]:
        """Extract and translate text from any file"""
        ext = filepath.suffix.lower()
        
        try:
            if ext == '.pdf':
                reader = PdfReader(str(filepath))
                raw_text = "\n".join([page.extract_text() for page in reader.pages])
            elif ext == '.docx':
                doc = Document(str(filepath))
                raw_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            elif ext == '.txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
            else:
                return "", "unsupported"
            
            original_lang = self.detect_language(raw_text)
            self.language_stats[original_lang] += 1
            
            # Translate if needed
            if original_lang in ['arabic', 'french']:
                print(f"  Translating from {original_lang}...", end=" ", flush=True)
                english_text = self.translate_to_english(raw_text, original_lang)
                print("✓")
            else:
                english_text = raw_text
            
            # Clean text
            english_text = re.sub(r'\n{3,}', '\n\n', english_text)
            english_text = re.sub(r'[^\x00-\x7F\u0080-\u00FF\s]', '', english_text)
            
            return english_text.strip(), original_lang
            
        except Exception as e:
            print(f"  Error: {e}")
            return "", "error"
    
    def process_documents(self):
        """Process all documents and save to JSON"""
        print("=" * 80)
        print("STEP 1: DOCUMENT EXTRACTION & TRANSLATION")
        print("=" * 80)
        
        doc_files = []
        for ext in ['.pdf', '.docx', '.txt']:
            doc_files.extend(self.docs_folder.glob(f'*{ext}'))
        
        if not doc_files:
            print(f"❌ No documents in {self.docs_folder}")
            return
        
        print(f"✓ Found {len(doc_files)} documents\n")
        
        processed_docs = []
        
        for idx, filepath in enumerate(doc_files, 1):
            print(f"[{idx}/{len(doc_files)}] {filepath.name[:70]}")
            
            english_text, original_lang = self.extract_text_from_file(filepath)
            
            if len(english_text) > 200:
                word_count = len(english_text.split())
                doc_data = {
                    'filename': filepath.name,
                    'text': english_text,
                    'original_lang': original_lang,
                    'word_count': word_count
                }
                processed_docs.append(doc_data)
                print(f"  ✓ {word_count:,} words | Language: {original_lang}")
            else:
                print(f"  ⚠ Skipped: insufficient content")
        
        # Save processed documents
        output_file = self.output_folder / "processed_documents.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': processed_docs,
                'stats': {
                    'total_documents': len(processed_docs),
                    'language_stats': self.language_stats,
                    'translation_stats': self.translation_stats
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✓ Processed {len(processed_docs)} documents")
        print(f"  Languages: {self.language_stats}")
        print(f"  Translated: AR={self.translation_stats['arabic_translated']}, FR={self.translation_stats['french_translated']}")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    processor = DocumentProcessor("Docs", "output")
    processor.process_documents()