"""
STEP 9: Finalize and Organize Datasets
✓ Creates clean final datasets with custom names
✓ English: final_data_eng.jsonl
✓ French: final_data_fr.jsonl
✓ Multilingual: final_data_multilingual.jsonl (optional)
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict
from collections import Counter


class DatasetFinalizer:
    def __init__(self, output_folder: str = "output"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
    
    def finalize_datasets(self, include_multilingual: bool = True):
        """
        Create final clean datasets with custom names
        
        Args:
            include_multilingual: Also create merged multilingual dataset
        """
        print("="*80)
        print("STEP 9: FINALIZE DATASETS WITH CUSTOM NAMES")
        print("="*80)
        print("Creating final datasets:")
        print("  - final_data_eng.jsonl (English only)")
        print("  - final_data_fr.jsonl (French only)")
        if include_multilingual:
            print("  - final_data_multilingual.jsonl (Both - optional)")
        print("="*80 + "\n")
        
        # Check if Step 8 prepared files exist
        prepared_en = self.output_folder / "prepared_data_english_20034pairs.jsonl"
        prepared_fr = self.output_folder / "prepared_data_french_20034pairs.jsonl"
        
        use_prepared = prepared_en.exists() and prepared_fr.exists()
        
        if use_prepared:
            print("✓ Found prepared datasets from Step 8\n")
        else:
            print("⚠ Step 8 prepared files not found, searching for augmented datasets...\n")
        
        # Find and process English dataset
        print("Processing English dataset...")
        if use_prepared:
            eng_dataset = self.load_jsonl_to_dataset(prepared_en)
            print(f"  ✓ Loaded from: {prepared_en.name}")
        else:
            eng_dataset = self.find_and_load_dataset('english')
        
        if eng_dataset:
            self.save_final_dataset(eng_dataset, 'eng', 'English')
        else:
            print("⚠ English dataset not found. Run augmentation first.")
        
        # Find and process French dataset
        print("\nProcessing French dataset...")
        if use_prepared:
            fr_dataset = self.load_jsonl_to_dataset(prepared_fr)
            print(f"  ✓ Loaded from: {prepared_fr.name}")
        else:
            fr_dataset = self.find_and_load_dataset('french')
        
        if fr_dataset:
            self.save_final_dataset(fr_dataset, 'fr', 'French')
        else:
            print("⚠ French dataset not found. Run translation & augmentation first.")
        
        # Create multilingual if requested
        if include_multilingual and eng_dataset and fr_dataset:
            print("\nCreating combined multilingual dataset...")
            self.create_multilingual_final(eng_dataset, fr_dataset)
        
        # Create summary
        self.create_final_summary()
    
    def find_and_load_dataset(self, language: str) -> List[Dict]:
        """Find best available dataset for language"""
        
        # Priority: HQ augmented > augmented > detailed > original
        candidates = []
        
        if language == 'english':
            candidates = [
                self.output_folder / "mistral_dataset_english_HQ_augmented_detailed.json",
                self.output_folder / "mistral_dataset_english_augmented_detailed.json",
                self.output_folder / "mistral_dataset_detailed.json",
            ]
            # Also check JSONL
            jsonl_candidates = [
                self.output_folder / "mistral_dataset_english_HQ_augmented_detailed.json",
                self.output_folder / "mistral_dataset_10017pairs.jsonl"
            ]
        else:  # french
            candidates = [
                self.output_folder / "mistral_dataset_french_HQ_augmented_detailed.json",
                self.output_folder / "mistral_dataset_french_augmented_detailed.json",
                self.output_folder / "mistral_dataset_french_detailed.json"
            ]
            jsonl_candidates = []
        
        # Try JSON files first
        for candidate in candidates:
            if candidate.exists():
                print(f"  ✓ Found: {candidate.name}")
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('dataset', data.get('qa_pairs', []))
        
        # Try JSONL files
        for candidate in jsonl_candidates:
            if candidate.exists():
                print(f"  ✓ Found: {candidate.name}")
                return self.load_jsonl_to_dataset(candidate)
        
        return None
    
    def load_jsonl_to_dataset(self, jsonl_file: Path) -> List[Dict]:
        """Load JSONL and convert to dataset format"""
        dataset = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = data['messages']
                qa = {
                    'instruction': messages[0]['content'],
                    'response': messages[1]['content'],
                    'category': 'general',
                    'difficulty': 'intermediate',
                    'source': 'original'
                }
                dataset.append(qa)
        return dataset
    
    def save_final_dataset(self, dataset: List[Dict], lang_code: str, lang_name: str):
        """Save final dataset with custom name"""
        
        # JSONL format (for training)
        jsonl_path = self.output_folder / f"final_data_{lang_code}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for qa in dataset:
                mistral_format = {
                    "messages": [
                        {"role": "user", "content": qa['instruction']},
                        {"role": "assistant", "content": qa['response']}
                    ]
                }
                f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
        
        print(f"  ✓ Created: final_data_{lang_code}.jsonl ({len(dataset):,} pairs)")
        
        # Detailed JSON (for analysis)
        detailed_path = self.output_folder / f"final_data_{lang_code}_detailed.json"
        
        # Calculate statistics
        stats = self.calculate_stats(dataset, lang_name)
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": stats,
                "dataset": dataset
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Created: final_data_{lang_code}_detailed.json")
        
        # Print statistics
        print(f"\n  {lang_name} Dataset Statistics:")
        print(f"  {'─'*76}")
        print(f"  Total pairs: {stats['total_pairs']:,}")
        print(f"  Avg instruction: {stats['avg_instruction_words']} words")
        print(f"  Avg response: {stats['avg_response_words']} words")
        
        if 'augmentation_types' in stats:
            print(f"\n  Composition:")
            for atype, count in sorted(stats['augmentation_types'].items(), key=lambda x: -x[1])[:5]:
                pct = (count / stats['total_pairs']) * 100
                print(f"    {atype}: {count:,} ({pct:.1f}%)")
    
    def create_multilingual_final(self, eng_dataset: List[Dict], fr_dataset: List[Dict]):
        """Create balanced multilingual dataset"""
        
        # Tag with language
        for qa in eng_dataset:
            qa['language'] = 'en'
        for qa in fr_dataset:
            qa['language'] = 'fr'
        
        # Combine and shuffle
        import random
        combined = eng_dataset + fr_dataset
        random.seed(42)  # Reproducible
        random.shuffle(combined)
        
        # Save
        jsonl_path = self.output_folder / "final_data_multilingual.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for qa in combined:
                mistral_format = {
                    "messages": [
                        {"role": "user", "content": qa['instruction']},
                        {"role": "assistant", "content": qa['response']}
                    ]
                }
                f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
        
        print(f"  ✓ Created: final_data_multilingual.jsonl ({len(combined):,} pairs)")
        
        # Detailed
        detailed_path = self.output_folder / "final_data_multilingual_detailed.json"
        stats = {
            "total_pairs": len(combined),
            "languages": {
                "en": len(eng_dataset),
                "fr": len(fr_dataset)
            },
            "balance": {
                "en_percent": (len(eng_dataset) / len(combined)) * 100,
                "fr_percent": (len(fr_dataset) / len(combined)) * 100
            }
        }
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": stats,
                "dataset": combined
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Created: final_data_multilingual_detailed.json")
        print(f"\n  Multilingual Statistics:")
        print(f"  {'─'*76}")
        print(f"  Total: {len(combined):,} pairs")
        print(f"  English: {len(eng_dataset):,} ({stats['balance']['en_percent']:.1f}%)")
        print(f"  French: {len(fr_dataset):,} ({stats['balance']['fr_percent']:.1f}%)")
    
    def calculate_stats(self, dataset: List[Dict], language: str) -> Dict:
        """Calculate dataset statistics"""
        
        # Length stats
        inst_words = [len(qa['instruction'].split()) for qa in dataset]
        resp_words = [len(qa['response'].split()) for qa in dataset]
        
        stats = {
            "language": language,
            "total_pairs": len(dataset),
            "avg_instruction_words": sum(inst_words) // len(inst_words),
            "avg_response_words": sum(resp_words) // len(resp_words),
            "min_instruction_words": min(inst_words),
            "max_instruction_words": max(inst_words),
            "min_response_words": min(resp_words),
            "max_response_words": max(resp_words)
        }
        
        # Category distribution
        if any('category' in qa for qa in dataset):
            categories = Counter([qa.get('category', 'unknown') for qa in dataset])
            stats['categories'] = dict(categories)
        
        # Difficulty distribution
        if any('difficulty' in qa for qa in dataset):
            difficulties = Counter([qa.get('difficulty', 'unknown') for qa in dataset])
            stats['difficulties'] = dict(difficulties)
        
        # Augmentation types
        if any('augmentation_type' in qa for qa in dataset):
            aug_types = Counter([qa.get('augmentation_type', 'original') for qa in dataset])
            stats['augmentation_types'] = dict(aug_types)
        
        return stats
    
    def create_final_summary(self):
        """Create a summary file of all final datasets"""
        
        summary = {
            "created": "Final datasets ready for training",
            "files": {},
            "recommendations": {}
        }
        
        # Check what was created
        final_files = [
            ("final_data_eng.jsonl", "English dataset"),
            ("final_data_fr.jsonl", "French dataset"),
            ("final_data_multilingual.jsonl", "Multilingual dataset")
        ]
        
        for filename, description in final_files:
            filepath = self.output_folder / filename
            if filepath.exists():
                # Count lines
                with open(filepath, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                
                summary["files"][filename] = {
                    "description": description,
                    "pairs": count,
                    "path": str(filepath),
                    "size_mb": filepath.stat().st_size / (1024 * 1024)
                }
        
        # Add recommendations
        if "final_data_multilingual.jsonl" in summary["files"]:
            summary["recommendations"]["for_multilingual_model"] = "Use final_data_multilingual.jsonl"
        if "final_data_eng.jsonl" in summary["files"]:
            summary["recommendations"]["for_english_only"] = "Use final_data_eng.jsonl"
        if "final_data_fr.jsonl" in summary["files"]:
            summary["recommendations"]["for_french_only"] = "Use final_data_fr.jsonl"
        
        # Save summary
        summary_path = self.output_folder / "FINAL_DATASETS_SUMMARY.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("FINAL DATASETS SUMMARY")
        print(f"{'='*80}")
        
        for filename, info in summary["files"].items():
            print(f"\n✓ {filename}")
            print(f"  Description: {info['description']}")
            print(f"  Pairs: {info['pairs']:,}")
            print(f"  Size: {info['size_mb']:.2f} MB")
            print(f"  Path: {info['path']}")
        
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS FOR TRAINING")
        print(f"{'='*80}")
        for purpose, recommendation in summary["recommendations"].items():
            print(f"  {purpose}: {recommendation}")
        
        print(f"\n✓ Summary saved: {summary_path}")
        print(f"{'='*80}\n")
        
        # Create README
        self.create_readme()
    
    def create_readme(self):
        """Create a README for final datasets"""
        
        readme_content = """# Final Datasets for Mistral Fine-Tuning

## 📁 Files Overview

### For Training (Use These)
- **final_data_eng.jsonl** - English dataset for training
- **final_data_fr.jsonl** - French dataset for training  
- **final_data_multilingual.jsonl** - Combined EN+FR dataset (shuffled)

### For Analysis (Optional)
- **final_data_eng_detailed.json** - English with metadata
- **final_data_fr_detailed.json** - French with metadata
- **final_data_multilingual_detailed.json** - Multilingual with metadata

## 🎯 Which File to Use?

### Multilingual Model (EN + FR)
```bash
Use: final_data_multilingual.jsonl
```
Best for: Models that need to understand both English and French

### English-Only Model
```bash
Use: final_data_eng.jsonl
```
Best for: English-specific fine-tuning

### French-Only Model
```bash
Use: final_data_fr.jsonl
```
Best for: French-specific fine-tuning

## 📊 Dataset Format

Each line in JSONL files:
```json
{"messages": [{"role": "user", "content": "Question here"}, {"role": "assistant", "content": "Answer here"}]}
```

This is the **Mistral chat format** - ready for fine-tuning!

## 🚀 Fine-Tuning Commands

### Using Mistral API
```bash
# Upload dataset
mistral finetune upload final_data_multilingual.jsonl

# Start fine-tuning
mistral finetune start --file your-file-id --model mistral-small-latest
```

### Using Hugging Face
```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset('json', data_files='final_data_multilingual.jsonl')

# Use with Trainer...
```

## 📈 Training Recommendations

- **Batch size**: 4-8 (depending on GPU)
- **Learning rate**: 1e-5 to 5e-5
- **Epochs**: 2-3
- **Validation split**: Use last 10% for validation

## ✅ Quality Assurance

All datasets include:
- Professional language
- Technical accuracy
- Balanced augmentation
- Strict quality filtering (85%+ threshold)
- Deduplicated content

## 📞 Files Location

All files are in the `output/` folder of your project.

---

Generated by Multilingual High-Quality Pipeline
"""
        
        readme_path = self.output_folder / "FINAL_DATASETS_README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ README created: {readme_path}\n")


if __name__ == "__main__":
    import sys
    
    include_multilingual = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--no-multilingual':
            include_multilingual = False
    
    finalizer = DatasetFinalizer("output")
    finalizer.finalize_datasets(include_multilingual=include_multilingual)