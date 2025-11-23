"""
STEP 8: Merge Multilingual Datasets
✓ Combines English and French datasets
✓ Intelligent shuffling for balanced training
✓ Deduplication across languages
✓ Multiple output formats
✓ Comprehensive statistics
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict, Counter


class MultilingualMerger:
    def __init__(self, output_folder: str = "output"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
    
    def merge_datasets(self, include_augmented: bool = True, shuffle: bool = False):
        """
        Merge English and French datasets (KEPT SEPARATE)
        
        Args:
            include_augmented: Include augmented variations (recommended)
            shuffle: Shuffle the combined dataset (DEFAULT: False - keeps separate)
        """
        print("="*80)
        print("STEP 8: PREPARE MULTILINGUAL DATASETS")
        print("="*80)
        print(f"Include augmented: {include_augmented}")
        print(f"Keep languages separate: {not shuffle}")
        print("="*80 + "\n")
        
        all_datasets = []
        
        # ========================================================================
        # Load English dataset
        # ========================================================================
        print("Loading English dataset...")
        en_dataset = self.load_dataset('english', include_augmented)
        
        if en_dataset:
            # Tag with language
            for qa in en_dataset:
                qa['language'] = 'en'
            all_datasets.extend(en_dataset)
            print(f"✓ Loaded {len(en_dataset):,} English Q&A pairs\n")
        else:
            print("⚠ English dataset not found\n")
        
        # ========================================================================
        # Load French dataset
        # ========================================================================
        print("Loading French dataset...")
        fr_dataset = self.load_dataset('french', include_augmented)
        
        if fr_dataset:
            # Tag with language
            for qa in fr_dataset:
                qa['language'] = 'fr'
            all_datasets.extend(fr_dataset)
            print(f"✓ Loaded {len(fr_dataset):,} French Q&A pairs\n")
        else:
            print("⚠ French dataset not found\n")
        
        if not all_datasets:
            print("❌ No datasets found to merge!")
            print("\nMake sure you have run:")
            print("  1. English augmentation: python 7_high_quality_augmentation.py english 2")
            print("  2. French augmentation: python 7_high_quality_augmentation.py french 2")
            return
        
        print(f"Total pairs before processing: {len(all_datasets):,}\n")
        
        # ========================================================================
        # Smart deduplication (within same language only)
        # ========================================================================
        print("Deduplicating within languages...")
        deduplicated = self.smart_deduplicate_multilingual(all_datasets)
        removed = len(all_datasets) - len(deduplicated)
        print(f"✓ After deduplication: {len(deduplicated):,} pairs")
        if removed > 0:
            print(f"  Removed {removed:,} duplicate pairs ({removed/len(all_datasets)*100:.1f}%)\n")
        else:
            print(f"  No duplicates found\n")
        
        # ========================================================================
        # Shuffle for balanced training (OPTIONAL - Default: Keep Separate)
        # ========================================================================
        if shuffle:
            print("Shuffling dataset for mixed training...")
            random.seed(42)  # Reproducible shuffle
            random.shuffle(deduplicated)
            print("✓ Dataset shuffled (EN/FR mixed)\n")
        else:
            print("Keeping languages separate (EN first, then FR)")
            # Sort to keep languages grouped
            deduplicated.sort(key=lambda x: x.get('language', 'en'))
            print("✓ Languages kept separate for independent training\n")
        
        # ========================================================================
        # Calculate comprehensive statistics
        # ========================================================================
        print("Calculating statistics...")
        stats = self.calculate_comprehensive_stats(deduplicated, en_dataset, fr_dataset)
        print("✓ Statistics calculated\n")
        
        # ========================================================================
        # Save datasets (SEPARATE by language)
        # ========================================================================
        print("Saving datasets...")
        self.save_separate_datasets(deduplicated, stats, include_augmented, en_dataset, fr_dataset)
    
    def load_dataset(self, language: str, include_augmented: bool) -> List[Dict]:
        """Load dataset with priority order"""
        
        candidates = []
        
        if language == 'english':
            if include_augmented:
                candidates = [
                    "mistral_dataset_english_HQ_augmented_detailed.json",
                    "mistral_dataset_english_augmented_detailed.json",
                    "mistral_dataset_detailed.json"
                ]
            else:
                candidates = [
                    "mistral_dataset_detailed.json"
                ]
            # Also try JSONL as fallback
            jsonl_fallback = "mistral_dataset_10017pairs.jsonl"
        else:  # french
            if include_augmented:
                candidates = [
                    "mistral_dataset_french_HQ_augmented_detailed.json",
                    "mistral_dataset_french_augmented_detailed.json",
                    "mistral_dataset_french_detailed.json"
                ]
            else:
                candidates = [
                    "mistral_dataset_french_detailed.json"
                ]
            jsonl_fallback = None
        
        # Try JSON files
        for candidate_name in candidates:
            candidate_path = self.output_folder / candidate_name
            if candidate_path.exists():
                print(f"  Found: {candidate_name}")
                with open(candidate_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('dataset', data.get('qa_pairs', []))
        
        # Try JSONL fallback for English
        if jsonl_fallback and language == 'english':
            jsonl_path = self.output_folder / jsonl_fallback
            if jsonl_path.exists():
                print(f"  Found: {jsonl_fallback}")
                return self.load_jsonl_dataset(jsonl_path)
        
        return None
    
    def load_jsonl_dataset(self, jsonl_path: Path) -> List[Dict]:
        """Load JSONL and convert to dataset format"""
        dataset = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    messages = data['messages']
                    qa = {
                        'instruction': messages[0]['content'],
                        'response': messages[1]['content'],
                        'category': 'general',
                        'difficulty': 'intermediate',
                        'source': 'original',
                        'augmentation_type': 'original'
                    }
                    dataset.append(qa)
                except:
                    continue
        return dataset
    
    def smart_deduplicate_multilingual(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Deduplicate within same language only
        Keeps best quality (longest response) for duplicates
        """
        seen = {}
        
        for qa in qa_pairs:
            lang = qa.get('language', 'en')
            instruction = qa['instruction']
            
            # Create unique key: language + normalized instruction
            normalized = self.normalize_text(instruction)[:100]
            key = (lang, normalized)
            
            # Keep if new, or if better quality than existing
            if key not in seen:
                seen[key] = qa
            else:
                # Keep the one with longer response (higher quality)
                if len(qa['response']) > len(seen[key]['response']):
                    seen[key] = qa
        
        return list(seen.values())
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
        return text.strip()
    
    def calculate_comprehensive_stats(self, merged_dataset: List[Dict], 
                                     en_dataset: List[Dict], 
                                     fr_dataset: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        
        stats = {
            'total_pairs': len(merged_dataset),
            'languages': {},
            'length_stats': {},
            'quality_metrics': {},
            'composition': {}
        }
        
        # Language distribution
        lang_counts = Counter([qa.get('language', 'unknown') for qa in merged_dataset])
        stats['languages'] = dict(lang_counts)
        
        # Calculate percentages
        stats['language_percentages'] = {
            lang: (count / len(merged_dataset)) * 100 
            for lang, count in lang_counts.items()
        }
        
        # Length statistics by language
        for lang in ['en', 'fr']:
            lang_pairs = [qa for qa in merged_dataset if qa.get('language') == lang]
            
            if lang_pairs:
                inst_words = [len(qa['instruction'].split()) for qa in lang_pairs]
                resp_words = [len(qa['response'].split()) for qa in lang_pairs]
                resp_sentences = [qa['response'].count('.') for qa in lang_pairs]
                
                stats['length_stats'][lang] = {
                    'count': len(lang_pairs),
                    'avg_instruction_words': sum(inst_words) // len(inst_words),
                    'avg_response_words': sum(resp_words) // len(resp_words),
                    'avg_response_sentences': sum(resp_sentences) / len(resp_sentences),
                    'min_response_words': min(resp_words),
                    'max_response_words': max(resp_words)
                }
        
        # Category distribution (if available)
        if any('category' in qa for qa in merged_dataset):
            categories = Counter([qa.get('category', 'unknown') for qa in merged_dataset])
            stats['categories'] = dict(categories)
        
        # Difficulty distribution (if available)
        if any('difficulty' in qa for qa in merged_dataset):
            difficulties = Counter([qa.get('difficulty', 'unknown') for qa in merged_dataset])
            stats['difficulties'] = dict(difficulties)
        
        # Augmentation types
        if any('augmentation_type' in qa for qa in merged_dataset):
            aug_types = Counter([
                qa.get('augmentation_type', 'original') 
                for qa in merged_dataset
            ])
            stats['augmentation_types'] = dict(aug_types)
            
            # Calculate augmentation ratio
            original_count = aug_types.get('original', 0)
            if original_count > 0:
                stats['augmentation_ratio'] = len(merged_dataset) / original_count
        
        # Quality metrics
        all_resp_words = [len(qa['response'].split()) for qa in merged_dataset]
        stats['quality_metrics'] = {
            'pairs_above_60_words': sum(1 for w in all_resp_words if w >= 60),
            'pairs_above_80_words': sum(1 for w in all_resp_words if w >= 80),
            'pairs_above_100_words': sum(1 for w in all_resp_words if w >= 100),
            'high_quality_percentage': (sum(1 for w in all_resp_words if w >= 60) / len(all_resp_words)) * 100
        }
        
        # Original dataset sizes
        stats['original_sizes'] = {
            'english': len(en_dataset) if en_dataset else 0,
            'french': len(fr_dataset) if fr_dataset else 0
        }
        
        return stats
    
    def save_separate_datasets(self, all_data: List[Dict], stats: Dict, 
                              augmented: bool, en_data: List[Dict], fr_data: List[Dict]):
        """Save datasets SEPARATELY by language"""
        
        suffix = "augmented" if augmented else "base"
        
        # Separate by language
        en_pairs = [qa for qa in all_data if qa.get('language') == 'en']
        fr_pairs = [qa for qa in all_data if qa.get('language') == 'fr']
        
        print(f"English pairs: {len(en_pairs):,}")
        print(f"French pairs: {len(fr_pairs):,}\n")
        
        # ========================================================================
        # Save ENGLISH dataset
        # ========================================================================
        if en_pairs:
            en_jsonl = self.output_folder / f"prepared_data_english_{len(en_pairs)}pairs.jsonl"
            with open(en_jsonl, 'w', encoding='utf-8') as f:
                for qa in en_pairs:
                    mistral_format = {
                        "messages": [
                            {"role": "user", "content": qa['instruction']},
                            {"role": "assistant", "content": qa['response']}
                        ]
                    }
                    f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
            
            print(f"✓ English JSONL: {en_jsonl}")
            print(f"  Size: {en_jsonl.stat().st_size / (1024*1024):.2f} MB")
        
        # ========================================================================
        # Save FRENCH dataset
        # ========================================================================
        if fr_pairs:
            fr_jsonl = self.output_folder / f"prepared_data_french_{len(fr_pairs)}pairs.jsonl"
            with open(fr_jsonl, 'w', encoding='utf-8') as f:
                for qa in fr_pairs:
                    mistral_format = {
                        "messages": [
                            {"role": "user", "content": qa['instruction']},
                            {"role": "assistant", "content": qa['response']}
                        ]
                    }
                    f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
            
            print(f"✓ French JSONL: {fr_jsonl}")
            print(f"  Size: {fr_jsonl.stat().st_size / (1024*1024):.2f} MB")
        
        # ========================================================================
        # Save COMBINED (optional - for reference)
        # ========================================================================
        combined_jsonl = self.output_folder / f"prepared_data_multilingual_{len(all_data)}pairs.jsonl"
        with open(combined_jsonl, 'w', encoding='utf-8') as f:
            for qa in all_data:
                mistral_format = {
                    "messages": [
                        {"role": "user", "content": qa['instruction']},
                        {"role": "assistant", "content": qa['response']}
                    ]
                }
                f.write(json.dumps(mistral_format, ensure_ascii=False) + '\n')
        
        print(f"✓ Combined JSONL: {combined_jsonl}")
        print(f"  Size: {combined_jsonl.stat().st_size / (1024*1024):.2f} MB\n")
        
        # ========================================================================
        # Save detailed JSON with metadata
        # ========================================================================
        detailed_path = self.output_folder / f"prepared_data_multilingual_detailed.json"
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": stats,
                "english_dataset": en_pairs,
                "french_dataset": fr_pairs
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Detailed JSON: {detailed_path}")
        print(f"  Size: {detailed_path.stat().st_size / (1024*1024):.2f} MB\n")
        
        # ========================================================================
        # Print comprehensive statistics
        # ========================================================================
        self.print_statistics(stats)
        
        # ========================================================================
        # Show samples
        # ========================================================================
        self.show_samples(all_data)
    
    def print_statistics(self, stats: Dict):
        """Print comprehensive statistics"""
        
        print(f"{'='*80}")
        print("MULTILINGUAL DATASET STATISTICS")
        print(f"{'='*80}")
        
        print(f"\n📊 OVERVIEW")
        print(f"{'─'*80}")
        print(f"Total Q&A pairs: {stats['total_pairs']:,}")
        
        print(f"\n🌍 LANGUAGE DISTRIBUTION")
        print(f"{'─'*80}")
        for lang, count in stats['languages'].items():
            pct = stats['language_percentages'][lang]
            bar = "█" * int(pct / 2)
            print(f"  {lang.upper()}: {count:,} ({pct:.1f}%) {bar}")
        
        print(f"\n📏 LENGTH STATISTICS")
        print(f"{'─'*80}")
        for lang, lstats in stats['length_stats'].items():
            print(f"  {lang.upper()}:")
            print(f"    Count: {lstats['count']:,} pairs")
            print(f"    Avg instruction: {lstats['avg_instruction_words']} words")
            print(f"    Avg response: {lstats['avg_response_words']} words")
            print(f"    Avg sentences: {lstats['avg_response_sentences']:.1f}")
            print(f"    Response range: {lstats['min_response_words']}-{lstats['max_response_words']} words")
        
        if 'augmentation_types' in stats:
            print(f"\n🔄 AUGMENTATION COMPOSITION")
            print(f"{'─'*80}")
            for atype, count in sorted(stats['augmentation_types'].items(), key=lambda x: -x[1]):
                pct = (count / stats['total_pairs']) * 100
                bar = "█" * int(pct / 2)
                print(f"  {atype}: {count:,} ({pct:.1f}%) {bar}")
            
            if 'augmentation_ratio' in stats:
                print(f"\n  Total expansion: {stats['augmentation_ratio']:.2f}x")
        
        if 'quality_metrics' in stats:
            print(f"\n✅ QUALITY METRICS")
            print(f"{'─'*80}")
            qm = stats['quality_metrics']
            print(f"  Responses ≥60 words: {qm['pairs_above_60_words']:,} ({qm['high_quality_percentage']:.1f}%)")
            print(f"  Responses ≥80 words: {qm['pairs_above_80_words']:,}")
            print(f"  Responses ≥100 words: {qm['pairs_above_100_words']:,}")
        
        if 'categories' in stats:
            print(f"\n🏷️  TOP CATEGORIES")
            print(f"{'─'*80}")
            for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1])[:10]:
                pct = (count / stats['total_pairs']) * 100
                print(f"  {cat}: {count:,} ({pct:.1f}%)")
        
        print(f"\n{'='*80}\n")
    
    def show_samples(self, dataset: List[Dict]):
        """Show sample Q&A pairs from each language"""
        
        print(f"{'='*80}")
        print("SAMPLE Q&A PAIRS")
        print(f"{'='*80}")
        
        for lang in ['en', 'fr']:
            lang_pairs = [qa for qa in dataset if qa.get('language') == lang]
            
            if lang_pairs:
                # Pick a random sample
                sample = random.choice(lang_pairs)
                
                lang_name = "ENGLISH" if lang == 'en' else "FRENCH"
                print(f"\n{lang_name} SAMPLE:")
                print(f"{'─'*80}")
                print(f"Category: {sample.get('category', 'N/A')}")
                print(f"Type: {sample.get('augmentation_type', 'original')}")
                print(f"\nQ: {sample['instruction']}")
                print(f"\nA: {sample['response'][:300]}...")
                print(f"\nStats: {len(sample['response'].split())} words, "
                      f"{sample['response'].count('.')} sentences")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    include_augmented = True
    shuffle = False  # DEFAULT: Keep separate
    
    if '--augmented' in sys.argv:
        include_augmented = True
    
    if '--no-augmented' in sys.argv:
        include_augmented = False
    
    if '--shuffle' in sys.argv:
        shuffle = True
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print("\nUsage:")
        print("  python 8_merge_multilingual.py                  # Separate EN & FR (recommended)")
        print("  python 8_merge_multilingual.py --no-augmented   # Use only original datasets")
        print("  python 8_merge_multilingual.py --shuffle        # Mix EN/FR together")
        sys.exit(0)
    
    merger = MultilingualMerger("output")
    merger.merge_datasets(include_augmented=include_augmented, shuffle=shuffle)