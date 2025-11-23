"""
STEP 6: Translate Dataset to French
✓ Translates English Q&A pairs to French
✓ Parallel processing for speed
✓ Checkpointing for resume capability
✓ Quality validation
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from deep_translator import GoogleTranslator
except ImportError:
    import os
    os.system("pip install deep-translator")
    from deep_translator import GoogleTranslator


class DatasetTranslator:
    def __init__(self, output_folder: str = "output", num_workers: int = 5):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        self.translator = GoogleTranslator(source='en', target='fr')
        self.num_workers = num_workers
        
        self.checkpoint_file = self.output_folder / "translation_checkpoint.json"
        self.stats = {
            "total": 0,
            "translated": 0,
            "failed": 0
        }
    
    def translate_text(self, text: str, max_retries: int = 3) -> str:
        """Translate text with retry logic"""
        for attempt in range(max_retries):
            try:
                # Split long texts into chunks (max 4500 chars)
                if len(text) > 4500:
                    chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
                    translated_chunks = []
                    for chunk in chunks:
                        translated = self.translator.translate(chunk)
                        translated_chunks.append(translated)
                        time.sleep(0.2)  # Rate limiting
                    return ' '.join(translated_chunks)
                else:
                    result = self.translator.translate(text)
                    time.sleep(0.2)  # Rate limiting
                    return result
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return text  # Return original if all retries fail
        
        return text
    
    def translate_qa_pair(self, qa: Dict, index: int) -> Dict:
        """Translate a single Q&A pair"""
        try:
            translated_qa = {
                'instruction': self.translate_text(qa['instruction']),
                'response': self.translate_text(qa['response']),
                'category': qa.get('category', 'general'),
                'difficulty': qa.get('difficulty', 'intermediate'),
                'source': qa.get('source', 'unknown'),
                'original_language': 'en',
                'translated_to': 'fr',
                'translation_index': index
            }
            
            # Validate translation quality
            if (len(translated_qa['instruction']) > 10 and 
                len(translated_qa['response']) > 50):
                return translated_qa
            else:
                return None
                
        except Exception as e:
            print(f"  ⚠ Translation error at index {index}: {e}")
            return None
    
    def load_checkpoint(self) -> Dict:
        """Load translation checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'translated_indices': [], 'french_qa_pairs': []}
    
    def save_checkpoint(self, translated_indices: List[int], french_qa: List[Dict]):
        """Save translation checkpoint"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'translated_indices': translated_indices,
                'french_qa_pairs': french_qa,
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats
            }, f, indent=2, ensure_ascii=False)
    
    def translate_dataset(self, resume: bool = False):
        """Translate entire dataset to French"""
        print("="*80)
        print("STEP 6: DATASET TRANSLATION (EN → FR)")
        print("="*80)
        print(f"Workers: {self.num_workers} | Rate limited translation")
        print("="*80 + "\n")
        
        # Load English dataset
        input_file = self.output_folder / "mistral_dataset_detailed.json"
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            print("Run steps 1-4 first!")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        english_qa = data['dataset']
        self.stats['total'] = len(english_qa)
        
        print(f"✓ Loaded {len(english_qa)} English Q&A pairs\n")
        
        # Load checkpoint
        checkpoint = self.load_checkpoint() if resume else {
            'translated_indices': [], 
            'french_qa_pairs': []
        }
        
        translated_indices = set(checkpoint['translated_indices'])
        french_qa_pairs = checkpoint['french_qa_pairs']
        
        if resume:
            print(f"✓ Resuming: {len(translated_indices)} already translated\n")
        
        # Prepare pending translations
        pending_qa = [
            (qa, idx) for idx, qa in enumerate(english_qa) 
            if idx not in translated_indices
        ]
        
        if not pending_qa:
            print("✅ All pairs already translated!")
            self.save_final_dataset(french_qa_pairs, data['metadata'])
            return
        
        print(f"🚀 Starting translation with {self.num_workers} workers...\n")
        
        start_time = time.time()
        completed = len(translated_indices)
        
        # Translate in parallel (using threads for I/O-bound task)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_qa = {
                executor.submit(self.translate_qa_pair, qa, idx): (qa, idx)
                for qa, idx in pending_qa
            }
            
            for future in as_completed(future_to_qa):
                qa, idx = future_to_qa[future]
                
                try:
                    translated = future.result()
                    
                    if translated:
                        french_qa_pairs.append(translated)
                        translated_indices.add(idx)
                        self.stats['translated'] += 1
                        completed += 1
                        
                        # Progress
                        progress = (completed / self.stats['total']) * 100
                        elapsed = time.time() - start_time
                        remaining = self.stats['total'] - completed
                        eta_seconds = (elapsed / completed) * remaining if completed > 0 else 0
                        
                        print(f"[{completed}/{self.stats['total']}] {progress:.1f}% | "
                              f"ETA: {eta_seconds/60:.1f} min | "
                              f"Source: {qa.get('source', 'unknown')[:40]}")
                    else:
                        self.stats['failed'] += 1
                        print(f"[{completed}/{self.stats['total']}] ⚠ Translation failed")
                    
                    # Save checkpoint every 20 translations
                    if completed % 20 == 0:
                        self.save_checkpoint(list(translated_indices), french_qa_pairs)
                        
                except Exception as e:
                    self.stats['failed'] += 1
                    print(f"[{completed}/{self.stats['total']}] ❌ Error: {str(e)[:50]}")
        
        # Final save
        self.save_checkpoint(list(translated_indices), french_qa_pairs)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✓ TRANSLATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Successfully translated: {self.stats['translated']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {(self.stats['translated']/self.stats['total']*100):.1f}%")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"{'='*80}\n")
        
        # Save final French dataset
        self.save_final_dataset(french_qa_pairs, data['metadata'])
    
    def save_final_dataset(self, french_qa: List[Dict], original_metadata: Dict):
        """Save French dataset in multiple formats"""
        
        # JSONL for Mistral fine-tuning
        jsonl_path = self.output_folder / f"mistral_dataset_french_{len(french_qa)}pairs.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for qa in french_qa:
                mistral_format = {
                    "messages": [
                        {"role": "user", "content": qa['instruction']},
                        {"role": "assistant", "content": qa['response']}
                    ]
                }
                f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
        
        print(f"✓ French JSONL saved: {jsonl_path}")
        
        # Detailed JSON
        detailed_path = self.output_folder / "mistral_dataset_french_detailed.json"
        french_metadata = {
            **original_metadata,
            "language": "french",
            "translated_from": "english",
            "total_pairs": len(french_qa)
        }
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": french_metadata,
                "dataset": french_qa
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ French detailed JSON saved: {detailed_path}")
        
        # Show sample
        print(f"\n{'='*80}")
        print("SAMPLE FRENCH Q&A")
        print(f"{'='*80}")
        if french_qa:
            sample = french_qa[min(5, len(french_qa)-1)]
            print(f"Category: {sample['category']} | Difficulty: {sample['difficulty']}")
            print(f"Source: {sample['source']}\n")
            print(f"Q: {sample['instruction']}\n")
            print(f"A: {sample['response'][:400]}...")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    translator = DatasetTranslator("output", num_workers=5)
    
    resume = len(sys.argv) > 1 and sys.argv[1] == '--resume'
    
    if resume:
        print("🔄 RESUMING translation from checkpoint...\n")
    
    translator.translate_dataset(resume=resume)