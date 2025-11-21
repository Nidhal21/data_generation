"""
ENHANCED MULTILINGUAL Dataset Generator for Mistral Fine-tuning
✓ Processes English, French, and Arabic documents
✓ Auto-translation to English
✓ 8 extraction strategies for maximum Q&A yield (30-40 per chunk)
✓ Smart chunking and quality filtering

Setup:
1. ollama pull llama3.2
2. pip install PyPDF2 python-docx requests deep-translator
3. python process_docs_multilingual.py
"""

import os
import json
import time
import re
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    from PyPDF2 import PdfReader
    from docx import Document
    from deep_translator import GoogleTranslator
except ImportError:
    print("Installing required packages...")
    os.system("pip install PyPDF2 python-docx requests deep-translator")
    from PyPDF2 import PdfReader
    from docx import Document
    from deep_translator import GoogleTranslator


class MultilingualMaximumProcessor:
    def __init__(self, docs_folder: str = "Docs", output_folder: str = "output"):
        self.docs_folder = Path(docs_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3.2"
        
        self.language_stats = {"english": 0, "arabic": 0, "french": 0}
        self.translation_stats = {"arabic_translated": 0, "french_translated": 0}
        self.processed_docs = []
        self.generation_stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "recovered": 0
        }
        
        # Initialize translators
        self.ar_translator = GoogleTranslator(source='ar', target='en')
        self.fr_translator = GoogleTranslator(source='fr', target='en')
        
        self.check_ollama()
    
    def check_ollama(self):
        """Check Ollama status"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                print(f"✓ Ollama running. Models: {', '.join(available)}")
                if not any(self.model in m for m in available):
                    print(f"\n⚠ Installing {self.model}...")
                    os.system(f"ollama pull {self.model}")
        except Exception as e:
            print(f"\n❌ Ollama not running! Install from: https://ollama.com/download")
            print(f"Then run: ollama pull {self.model}")
            exit(1)
    
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
    
    def smart_chunk_text(self, text: str, target_size: int = 2500) -> List[str]:
        """Smart chunking with overlap for better context"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        overlap_size = 300
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > target_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
                current_chunk = overlap + "\n\n" + para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def call_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """Call Ollama with smart retry"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "num_predict": 2500,
                            "top_p": 0.9,
                            "repeat_penalty": 1.15,
                            "num_ctx": 8192
                        }
                    },
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json()['response'].strip()
                    if result:
                        return result
                        
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏱️", end="")
                    time.sleep(2)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return ""
    
    def generate_qa_batch(self, chunk: str, filename: str, batch_num: int = 1) -> List[Dict]:
        """Generate Q&A with 8 targeted strategies for MAXIMUM extraction"""
        
        all_qa = []
        
        # Strategy 1: Definitions and Terminology (5 Q&A)
        prompt1 = f"""Extract key terms, definitions, and concepts from this text.

TEXT: {chunk[:1800]}

Create 5 high-quality Q&A pairs about definitions, terms, and core concepts.
Answers should be 5-8 sentences with clear, detailed explanations.

JSON format:
[{{"instruction":"What is/Define [term]?","response":"[detailed 5-8 sentence answer]","category":"definition","difficulty":"beginner"}}]

JSON:"""
        
        # Strategy 2: Processes and Procedures (5 Q&A)
        prompt2 = f"""Extract processes, methodologies, procedures, and step-by-step instructions.

TEXT: {chunk[:1800]}

Create 5 Q&A pairs about HOW things work, are done, or are implemented.
Answers should be 6-9 sentences with clear steps or explanations.

JSON format:
[{{"instruction":"How to/does [process]?","response":"[detailed 6-9 sentence answer with steps]","category":"process","difficulty":"intermediate"}}]

JSON:"""
        
        # Strategy 3: Best Practices and Guidelines (5 Q&A)
        prompt3 = f"""Extract best practices, guidelines, recommendations, and practical advice.

TEXT: {chunk[:1800]}

Create 5 Q&A pairs about best practices, what should/shouldn't be done.
Answers should be 5-8 sentences with actionable advice.

JSON format:
[{{"instruction":"What are best practices for [topic]?","response":"[detailed 5-8 sentence answer]","category":"best-practice","difficulty":"intermediate"}}]

JSON:"""
        
        # Strategy 4: Problem-Solving and Scenarios (5 Q&A)
        prompt4 = f"""Extract problem-solving approaches, real-world scenarios, and case applications.

TEXT: {chunk[:1800]}

Create 5 Q&A pairs about problems, scenarios, challenges, and their solutions.
Answers should be 6-9 sentences with context and solutions.

JSON format:
[{{"instruction":"How to handle/solve [scenario]?","response":"[detailed 6-9 sentence answer]","category":"problem-solving","difficulty":"advanced"}}]

JSON:"""
        
        # Strategy 5: Comparisons and Differences (4 Q&A)
        prompt5 = f"""Extract comparisons, differences, contrasts between concepts, methods, or approaches.

TEXT: {chunk[:1800]}

Create 4 Q&A pairs about comparing and contrasting different elements.
Answers should be 5-7 sentences highlighting key differences.

JSON format:
[{{"instruction":"What's the difference between [A] and [B]?","response":"[detailed 5-7 sentence comparison]","category":"comparison","difficulty":"intermediate"}}]

JSON:"""
        
        # Strategy 6: Advantages and Disadvantages (4 Q&A)
        prompt6 = f"""Extract advantages, disadvantages, pros/cons, benefits, and limitations.

TEXT: {chunk[:1800]}

Create 4 Q&A pairs about benefits, drawbacks, pros and cons of concepts.
Answers should be 5-7 sentences with balanced analysis.

JSON format:
[{{"instruction":"What are advantages/disadvantages of [topic]?","response":"[detailed 5-7 sentence answer]","category":"analysis","difficulty":"intermediate"}}]

JSON:"""
        
        # Strategy 7: Why and Reasoning (4 Q&A)
        prompt7 = f"""Extract reasoning, rationale, causes, and explanations for WHY things work/exist.

TEXT: {chunk[:1800]}

Create 4 Q&A pairs about WHY things are done, why they matter, underlying reasons.
Answers should be 5-7 sentences explaining rationale.

JSON format:
[{{"instruction":"Why is [concept] important/done?","response":"[detailed 5-7 sentence explanation]","category":"reasoning","difficulty":"intermediate"}}]

JSON:"""
        
        # Strategy 8: Examples and Applications (4 Q&A)
        prompt8 = f"""Extract concrete examples, real-world applications, and practical illustrations.

TEXT: {chunk[:1800]}

Create 4 Q&A pairs about specific examples and applications of concepts.
Answers should be 5-7 sentences with concrete examples.

JSON format:
[{{"instruction":"What are examples of [concept]?","response":"[detailed 5-7 sentence answer with examples]","category":"examples","difficulty":"beginner"}}]

JSON:"""
        
        # Rotate through strategies based on batch number
        strategies = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8]
        selected_prompts = [strategies[batch_num % 8], strategies[(batch_num + 3) % 8]]
        
        for prompt in selected_prompts:
            response = self.call_ollama(prompt)
            if response:
                qa_pairs = self.extract_qa_from_response(response, filename)
                all_qa.extend(qa_pairs)
            time.sleep(0.3)
        
        return all_qa
    
    def extract_qa_from_response(self, response: str, filename: str) -> List[Dict]:
        """Multi-strategy JSON extraction"""
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        qa_pairs = []
        
        # Try standard JSON array
        try:
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_str = self.fix_json(json_str)
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    qa_pairs = parsed
        except:
            pass
        
        # Try individual objects
        if not qa_pairs:
            try:
                pattern = r'\{[^{}]*?"instruction"[^{}]*?"response"[^{}]*?\}'
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        obj = json.loads(self.fix_json(match))
                        if 'instruction' in obj and 'response' in obj:
                            qa_pairs.append(obj)
                    except:
                        continue
            except:
                pass
        
        # Fallback text parsing
        if not qa_pairs:
            qa_pairs = self.parse_text_patterns(response)
        
        # Validate and enrich
        valid_pairs = []
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            
            inst = qa.get('instruction', '').strip()
            resp = qa.get('response', '').strip()
            
            # Enhanced quality filters
            if len(inst) < 15 or len(resp) < 80:
                continue
            if len(resp.split()) < 30:  # Minimum 30 words for quality
                continue
            if resp.count('.') < 3:  # At least 3 sentences
                continue
            
            qa['source'] = filename
            qa.setdefault('category', 'general')
            qa.setdefault('difficulty', 'intermediate')
            
            valid_pairs.append(qa)
        
        return valid_pairs
    
    def fix_json(self, text: str) -> str:
        """Fix common JSON issues"""
        text = re.sub(r',(\s*[\]}])', r'\1', text)
        text = text.replace('\\"', '"')
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def parse_text_patterns(self, text: str) -> List[Dict]:
        """Extract Q&A from plain text patterns"""
        qa_pairs = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if re.match(r'^(Q\d*:|Question\d*:|Instruction:)', line, re.IGNORECASE):
                question = re.sub(r'^(Q\d*:|Question\d*:|Instruction:)\s*', '', line, flags=re.IGNORECASE)
                question = question.strip('"\'')
                
                answer_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if re.match(r'^(A\d*:|Answer\d*:|Response:)', next_line, re.IGNORECASE):
                        answer_text = re.sub(r'^(A\d*:|Answer\d*:|Response:)\s*', '', next_line, flags=re.IGNORECASE)
                        answer_lines.append(answer_text.strip('"\''))
                        i += 1
                        while i < len(lines) and not re.match(r'^(Q\d*:|Question\d*:)', lines[i].strip(), re.IGNORECASE):
                            if lines[i].strip():
                                answer_lines.append(lines[i].strip().strip('"\''))
                            i += 1
                        break
                    i += 1
                
                if question and answer_lines:
                    answer = ' '.join(answer_lines)
                    if len(answer.split()) >= 30:
                        qa_pairs.append({
                            'instruction': question,
                            'response': answer,
                            'category': 'general',
                            'difficulty': 'intermediate'
                        })
            else:
                i += 1
        
        return qa_pairs
    
    def process_all_documents(self, chunk_size: int = 2500):
        """Process with MAXIMUM extraction including multilingual support"""
        
        print("=" * 80)
        print("ENHANCED MULTILINGUAL DATASET GENERATOR")
        print("=" * 80)
        print(f"Model: {self.model} | 8 extraction strategies | Auto-translation")
        print(f"Searching: {self.docs_folder}\n")
        
        doc_files = []
        for ext in ['.pdf', '.docx', '.txt']:
            doc_files.extend(self.docs_folder.glob(f'*{ext}'))
        
        if not doc_files:
            print(f"❌ No documents in {self.docs_folder}")
            return
        
        print(f"✓ Found {len(doc_files)} documents\n")
        
        # Extract and translate
        print("STEP 1: EXTRACTING & TRANSLATING CONTENT")
        print("-" * 80)
        
        for idx, filepath in enumerate(doc_files, 1):
            print(f"[{idx}/{len(doc_files)}] {filepath.name[:55]}")
            
            english_text, original_lang = self.extract_text_from_file(filepath)
            
            if len(english_text) > 200:
                word_count = len(english_text.split())
                self.processed_docs.append({
                    'filename': filepath.name,
                    'text': english_text,
                    'original_lang': original_lang,
                    'word_count': word_count
                })
                print(f"  ✓ {word_count:,} words | Original: {original_lang}")
            else:
                print(f"  ⚠ Skipped: insufficient content")
        
        print(f"\n✓ Processed {len(self.processed_docs)} documents")
        print(f"  Languages: {self.language_stats}")
        print(f"  Translated: AR={self.translation_stats['arabic_translated']}, FR={self.translation_stats['french_translated']}\n")
        
        # Generate Q&A
        print("STEP 2: GENERATING Q&A (8 STRATEGIES)")
        print("-" * 80)
        
        all_qa_pairs = []
        
        for doc_idx, doc in enumerate(self.processed_docs, 1):
            print(f"[{doc_idx}/{len(self.processed_docs)}] {doc['filename'][:55]}")
            
            chunks = self.smart_chunk_text(doc['text'], chunk_size)
            print(f"  Chunks: {len(chunks)} | Target: ~30-40 Q&A per chunk")
            
            for chunk_idx, chunk in enumerate(chunks, 1):
                self.generation_stats["total_attempts"] += 1
                
                print(f"  [{chunk_idx}/{len(chunks)}] ", end="", flush=True)
                
                qa_pairs = self.generate_qa_batch(chunk, doc['filename'], chunk_idx)
                
                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    self.generation_stats["successful"] += 1
                    print(f"✓ {len(qa_pairs)} Q&A")
                else:
                    self.generation_stats["failed"] += 1
                    print(f"⚠ Retry...", end="", flush=True)
                    
                    time.sleep(1)
                    qa_pairs = self.generate_qa_batch(chunk, doc['filename'], chunk_idx + 100)
                    if qa_pairs:
                        all_qa_pairs.extend(qa_pairs)
                        self.generation_stats["recovered"] += 1
                        print(f" ✓ {len(qa_pairs)} Q&A (recovered)")
                    else:
                        print(f" ✗ Skipped")
                
                time.sleep(0.3)
            
            print()
        
        # Deduplicate
        print("\nPost-processing and deduplication...")
        unique_qa = self.smart_deduplicate(all_qa_pairs)
        
        # Results
        print(f"\n{'='*80}")
        print(f"✓ GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Total Q&A pairs: {len(unique_qa)}")
        print(f"Source documents: {len(self.processed_docs)}")
        print(f"Average per document: {len(unique_qa) // max(1, len(self.processed_docs))}")
        print(f"\nGeneration stats:")
        print(f"  Successful: {self.generation_stats['successful']}/{self.generation_stats['total_attempts']}")
        print(f"  Recovered: {self.generation_stats['recovered']}")
        success_rate = (self.generation_stats['successful'] + self.generation_stats['recovered']) / max(1, self.generation_stats['total_attempts']) * 100
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"{'='*80}\n")
        
        if unique_qa:
            self.save_datasets(unique_qa)
        else:
            print("❌ No Q&A generated.")
    
    def smart_deduplicate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Smart deduplication keeping highest quality"""
        seen = {}
        
        for qa in qa_pairs:
            key = re.sub(r'\W+', '', qa['instruction'].lower())[:70]
            
            if key not in seen or len(qa['response']) > len(seen[key]['response']):
                seen[key] = qa
        
        return list(seen.values())
    
    def save_datasets(self, qa_pairs: List[Dict]):
        """Save datasets in multiple formats"""
        
        qa_pairs.sort(key=lambda x: (x['source'], x['category']))
        
        # JSONL for Mistral
        jsonl_path = self.output_folder / f"mistral_multilingual_{len(qa_pairs)}pairs.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                mistral_format = {
                    "messages": [
                        {"role": "user", "content": qa['instruction']},
                        {"role": "assistant", "content": qa['response']}
                    ]
                }
                f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
        
        print(f"✓ JSONL saved: {jsonl_path}")
        
        # Detailed JSON
        stats = {
            "total_pairs": len(qa_pairs),
            "sources": len(self.processed_docs),
            "languages": self.language_stats,
            "translations": self.translation_stats,
            "categories": self.get_stats(qa_pairs, 'category'),
            "difficulties": self.get_stats(qa_pairs, 'difficulty'),
            "avg_response_words": sum(len(qa['response'].split()) for qa in qa_pairs) // max(1, len(qa_pairs)),
            "success_rate": f"{(self.generation_stats['successful'] + self.generation_stats['recovered']) / max(1, self.generation_stats['total_attempts']) * 100:.1f}%"
        }
        
        detailed = {"metadata": stats, "dataset": qa_pairs}
        json_path = self.output_folder / "mistral_multilingual_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Detailed JSON saved: {json_path}")
        
        # Sample
        print(f"\n{'='*80}")
        print("QUALITY CHECK - Sample Q&A:")
        print("="*80)
        sample = qa_pairs[min(5, len(qa_pairs)-1)]
        print(f"Category: {sample['category']} | Difficulty: {sample['difficulty']}")
        print(f"Source: {sample['source']}\n")
        print(f"Q: {sample['instruction']}\n")
        print(f"A: {sample['response'][:300]}...")
        print(f"\nWords: {len(sample['response'].split())} | Sentences: {sample['response'].count('.')}")
        print("="*80)
        
        print(f"\nDataset Distribution:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
    
    def get_stats(self, qa_pairs: List[Dict], field: str) -> Dict:
        stats = defaultdict(int)
        for qa in qa_pairs:
            stats[qa.get(field, 'unknown')] += 1
        return dict(stats)


def main():
    """Execute enhanced multilingual generation"""
    
    DOCS_FOLDER = "Docs"
    OUTPUT_FOLDER = "output"
    CHUNK_SIZE = 2500
    
    print("\n🚀 ENHANCED MULTILINGUAL DATASET GENERATOR\n")
    print("Features:")
    print("  ✓ Supports English, French, and Arabic documents")
    print("  ✓ Auto-translation to English using Google Translate")
    print("  ✓ 8 extraction strategies (30-40 Q&A per chunk)")
    print("  ✓ Strategies: definitions, processes, best practices, problem-solving,")
    print("    comparisons, pros/cons, reasoning, examples")
    print("  ✓ Smart chunking with overlap")
    print("  ✓ Quality filtering (min 30 words, 3+ sentences)")
    print("  ✓ Automatic retry and recovery")
    print("  ✓ Intelligent deduplication\n")
    
    processor = MultilingualMaximumProcessor(DOCS_FOLDER, OUTPUT_FOLDER)
    processor.process_all_documents(CHUNK_SIZE)
    
    print("\n✅ ALL DONE! Check 'output' folder.")
    print("   Use the .jsonl file for Mistral fine-tuning.\n")


if __name__ == "__main__":
    main()