"""
STEP 4: Dataset Finalization
Deduplicates, validates, and formats data for Mistral fine-tuning
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


class DatasetFinalizer:
    def __init__(self, output_folder: str = "output"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
    
    def smart_deduplicate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Smart deduplication keeping highest quality"""
        seen = {}
        
        for qa in qa_pairs:
            # Create key from instruction
            key = re.sub(r'\W+', '', qa['instruction'].lower())[:70]
            
            # Keep if new or better quality (longer response)
            if key not in seen or len(qa['response']) > len(seen[key]['response']):
                seen[key] = qa
        
        return list(seen.values())
    
    def validate_qa_pair(self, qa: Dict) -> bool:
        """Validate Q&A pair quality"""
        try:
            inst = qa.get('instruction', '')
            resp = qa.get('response', '')
            
            # Basic validation
            if not inst or not resp:
                return False
            
            # Length checks
            if len(inst) < 15 or len(resp) < 80:
                return False
            
            # Word count
            if len(resp.split()) < 30:
                return False
            
            # Sentence count
            if resp.count('.') < 3:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_stats(self, qa_pairs: List[Dict], field: str) -> Dict:
        """Get distribution statistics"""
        stats = defaultdict(int)
        for qa in qa_pairs:
            stats[qa.get(field, 'unknown')] += 1
        return dict(stats)
    
    def finalize_dataset(self):
        """Finalize dataset for Mistral training"""
        print("=" * 80)
        print("STEP 4: DATASET FINALIZATION")
        print("=" * 80)
        
        # Load raw Q&A pairs
        input_file = self.output_folder / "raw_qa_pairs.json"
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            print("Run 3_qa_generator.py first!")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_qa_pairs = data['qa_pairs']
        print(f"✓ Loaded {len(raw_qa_pairs)} raw Q&A pairs\n")
        
        # Validate
        print("Validating Q&A pairs...")
        valid_qa = [qa for qa in raw_qa_pairs if self.validate_qa_pair(qa)]
        print(f"  ✓ {len(valid_qa)} valid pairs (removed {len(raw_qa_pairs) - len(valid_qa)} invalid)")
        
        # Deduplicate
        print("Deduplicating...")
        unique_qa = self.smart_deduplicate(valid_qa)
        print(f"  ✓ {len(unique_qa)} unique pairs (removed {len(valid_qa) - len(unique_qa)} duplicates)")
        
        # Sort by source and category
        unique_qa.sort(key=lambda x: (x.get('source', ''), x.get('category', '')))
        
        # Calculate statistics
        stats = {
            "total_pairs": len(unique_qa),
            "categories": self.get_stats(unique_qa, 'category'),
            "difficulties": self.get_stats(unique_qa, 'difficulty'),
            "sources": self.get_stats(unique_qa, 'source'),
            "avg_instruction_words": sum(len(qa['instruction'].split()) for qa in unique_qa) // max(1, len(unique_qa)),
            "avg_response_words": sum(len(qa['response'].split()) for qa in unique_qa) // max(1, len(unique_qa)),
        }
        
        print(f"\n{'='*80}")
        print("FINAL DATASET STATISTICS")
        print(f"{'='*80}")
        print(f"Total Q&A pairs: {stats['total_pairs']}")
        print(f"Average instruction length: {stats['avg_instruction_words']} words")
        print(f"Average response length: {stats['avg_response_words']} words")
        print(f"\nCategories:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
        print(f"\nDifficulties:")
        for diff, count in sorted(stats['difficulties'].items(), key=lambda x: -x[1]):
            print(f"  {diff}: {count}")
        print(f"{'='*80}\n")
        
        # Save JSONL for Mistral fine-tuning
        jsonl_path = self.output_folder / f"mistral_dataset_{len(unique_qa)}pairs.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for qa in unique_qa:
                mistral_format = {
                    "messages": [
                        {"role": "user", "content": qa['instruction']},
                        {"role": "assistant", "content": qa['response']}
                    ]
                }
                f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
        
        print(f"✓ Mistral JSONL saved: {jsonl_path}")
        
        # Save detailed JSON with metadata
        detailed_path = self.output_folder / "mistral_dataset_detailed.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": stats,
                "dataset": unique_qa
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Detailed JSON saved: {detailed_path}")
        
        # Show sample
        print(f"\n{'='*80}")
        print("SAMPLE Q&A PAIR")
        print(f"{'='*80}")
        if unique_qa:
            sample = unique_qa[min(5, len(unique_qa)-1)]
            print(f"Category: {sample['category']} | Difficulty: {sample['difficulty']}")
            print(f"Source: {sample['source']}\n")
            print(f"Q: {sample['instruction']}\n")
            print(f"A: {sample['response'][:400]}...")
            print(f"\nWords: {len(sample['response'].split())} | Sentences: {sample['response'].count('.')}")
        print(f"{'='*80}\n")
        
        print("✅ DATASET READY FOR MISTRAL FINE-TUNING!")
        print(f"   Use file: {jsonl_path}\n")


if __name__ == "__main__":
    finalizer = DatasetFinalizer("output")
    finalizer.finalize_dataset()