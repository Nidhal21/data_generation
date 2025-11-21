"""
Dataset Analysis Tool
Analyzes the quality and structure of generated dataset
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import statistics


class DatasetAnalyzer:
    def __init__(self, output_folder: str = "output"):
        self.output_folder = Path(output_folder)
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        
        # Find the detailed JSON file
        json_files = list(self.output_folder.glob("mistral_dataset_detailed.json"))
        if not json_files:
            json_files = list(self.output_folder.glob("raw_qa_pairs.json"))
        
        if not json_files:
            print("❌ No dataset found in output folder")
            return
        
        dataset_file = json_files[0]
        print(f"Analyzing: {dataset_file.name}\n")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract Q&A pairs
        if 'dataset' in data:
            qa_pairs = data['dataset']
            metadata = data.get('metadata', {})
        elif 'qa_pairs' in data:
            qa_pairs = data['qa_pairs']
            metadata = {}
        else:
            print("❌ Invalid dataset format")
            return
        
        if not qa_pairs:
            print("❌ No Q&A pairs found")
            return
        
        print("="*80)
        print("DATASET QUALITY ANALYSIS")
        print("="*80)
        
        # Basic stats
        print(f"\n📊 OVERVIEW")
        print(f"{'─'*80}")
        print(f"Total Q&A pairs: {len(qa_pairs)}")
        
        # Length analysis
        instruction_lengths = [len(qa['instruction']) for qa in qa_pairs]
        response_lengths = [len(qa['response']) for qa in qa_pairs]
        instruction_words = [len(qa['instruction'].split()) for qa in qa_pairs]
        response_words = [len(qa['response'].split()) for qa in qa_pairs]
        response_sentences = [qa['response'].count('.') for qa in qa_pairs]
        
        print(f"\n📝 INSTRUCTION ANALYSIS")
        print(f"{'─'*80}")
        print(f"Average length: {statistics.mean(instruction_lengths):.0f} chars ({statistics.mean(instruction_words):.0f} words)")
        print(f"Shortest: {min(instruction_lengths)} chars ({min(instruction_words)} words)")
        print(f"Longest: {max(instruction_lengths)} chars ({max(instruction_words)} words)")
        print(f"Median: {statistics.median(instruction_lengths):.0f} chars")
        
        print(f"\n💬 RESPONSE ANALYSIS")
        print(f"{'─'*80}")
        print(f"Average length: {statistics.mean(response_lengths):.0f} chars ({statistics.mean(response_words):.0f} words)")
        print(f"Average sentences: {statistics.mean(response_sentences):.1f}")
        print(f"Shortest: {min(response_lengths)} chars ({min(response_words)} words)")
        print(f"Longest: {max(response_lengths)} chars ({max(response_words)} words)")
        print(f"Median: {statistics.median(response_lengths):.0f} chars")
        
        # Distribution analysis
        print(f"\n📈 LENGTH DISTRIBUTION")
        print(f"{'─'*80}")
        
        # Response word count distribution
        word_ranges = [
            (0, 30, "Very Short (<30 words)"),
            (30, 50, "Short (30-50 words)"),
            (50, 80, "Medium (50-80 words)"),
            (80, 120, "Long (80-120 words)"),
            (120, 1000, "Very Long (>120 words)")
        ]
        
        for min_w, max_w, label in word_ranges:
            count = sum(1 for w in response_words if min_w <= w < max_w)
            pct = (count / len(response_words)) * 100
            bar = "█" * int(pct / 2)
            print(f"{label:25} {count:4} ({pct:5.1f}%) {bar}")
        
        # Sentence distribution
        print(f"\n📄 SENTENCE DISTRIBUTION")
        print(f"{'─'*80}")
        sentence_ranges = [
            (0, 3, "Few (<3 sentences)"),
            (3, 5, "Moderate (3-5 sentences)"),
            (5, 8, "Good (5-8 sentences)"),
            (8, 12, "Detailed (8-12 sentences)"),
            (12, 100, "Very Detailed (>12 sentences)")
        ]
        
        for min_s, max_s, label in sentence_ranges:
            count = sum(1 for s in response_sentences if min_s <= s < max_s)
            pct = (count / len(response_sentences)) * 100
            bar = "█" * int(pct / 2)
            print(f"{label:30} {count:4} ({pct:5.1f}%) {bar}")
        
        # Category analysis
        if any('category' in qa for qa in qa_pairs):
            print(f"\n🏷️  CATEGORY DISTRIBUTION")
            print(f"{'─'*80}")
            categories = [qa.get('category', 'unknown') for qa in qa_pairs]
            cat_counts = Counter(categories)
            for cat, count in cat_counts.most_common():
                pct = (count / len(qa_pairs)) * 100
                bar = "█" * int(pct / 2)
                print(f"{cat:25} {count:4} ({pct:5.1f}%) {bar}")
        
        # Difficulty analysis
        if any('difficulty' in qa for qa in qa_pairs):
            print(f"\n🎯 DIFFICULTY DISTRIBUTION")
            print(f"{'─'*80}")
            difficulties = [qa.get('difficulty', 'unknown') for qa in qa_pairs]
            diff_counts = Counter(difficulties)
            for diff, count in diff_counts.most_common():
                pct = (count / len(qa_pairs)) * 100
                bar = "█" * int(pct / 2)
                print(f"{diff:25} {count:4} ({pct:5.1f}%) {bar}")
        
        # Source analysis
        if any('source' in qa for qa in qa_pairs):
            print(f"\n📚 SOURCE DOCUMENTS")
            print(f"{'─'*80}")
            sources = [qa.get('source', 'unknown') for qa in qa_pairs]
            source_counts = Counter(sources)
            print(f"Total documents: {len(source_counts)}")
            print(f"\nTop 10 sources:")
            for source, count in source_counts.most_common(10):
                avg_per_doc = count
                print(f"  {source[:60]:60} {count:4} Q&A")
        
        # Quality indicators
        print(f"\n✅ QUALITY INDICATORS")
        print(f"{'─'*80}")
        
        # High quality (>80 words, >5 sentences)
        high_quality = sum(1 for i in range(len(qa_pairs)) 
                          if response_words[i] >= 80 and response_sentences[i] >= 5)
        print(f"High quality (80+ words, 5+ sentences): {high_quality} ({high_quality/len(qa_pairs)*100:.1f}%)")
        
        # Medium quality (50-80 words, 3-5 sentences)
        medium_quality = sum(1 for i in range(len(qa_pairs)) 
                            if 50 <= response_words[i] < 80 and 3 <= response_sentences[i] < 5)
        print(f"Medium quality (50-80 words, 3-5 sent): {medium_quality} ({medium_quality/len(qa_pairs)*100:.1f}%)")
        
        # Low quality (<50 words or <3 sentences)
        low_quality = sum(1 for i in range(len(qa_pairs)) 
                         if response_words[i] < 50 or response_sentences[i] < 3)
        print(f"Low quality (<50 words or <3 sent):    {low_quality} ({low_quality/len(qa_pairs)*100:.1f}%)")
        
        # Question type analysis
        print(f"\n❓ QUESTION PATTERNS")
        print(f"{'─'*80}")
        
        question_starters = defaultdict(int)
        for qa in qa_pairs:
            inst = qa['instruction'].lower()
            if inst.startswith('what'):
                question_starters['What'] += 1
            elif inst.startswith('how'):
                question_starters['How'] += 1
            elif inst.startswith('why'):
                question_starters['Why'] += 1
            elif inst.startswith('when'):
                question_starters['When'] += 1
            elif inst.startswith('where'):
                question_starters['Where'] += 1
            elif 'define' in inst[:20]:
                question_starters['Define'] += 1
            elif 'explain' in inst[:20]:
                question_starters['Explain'] += 1
            elif 'describe' in inst[:20]:
                question_starters['Describe'] += 1
            else:
                question_starters['Other'] += 1
        
        for qtype, count in sorted(question_starters.items(), key=lambda x: -x[1]):
            pct = (count / len(qa_pairs)) * 100
            bar = "█" * int(pct / 2)
            print(f"{qtype:15} {count:4} ({pct:5.1f}%) {bar}")
        
        # Sample high-quality Q&A
        print(f"\n{'='*80}")
        print("SAMPLE HIGH-QUALITY Q&A")
        print(f"{'='*80}")
        
        # Find longest, most detailed response
        best_idx = max(range(len(qa_pairs)), 
                      key=lambda i: response_words[i] + response_sentences[i])
        best_qa = qa_pairs[best_idx]
        
        print(f"\nCategory: {best_qa.get('category', 'N/A')} | Difficulty: {best_qa.get('difficulty', 'N/A')}")
        print(f"Source: {best_qa.get('source', 'N/A')}")
        print(f"Stats: {response_words[best_idx]} words, {response_sentences[best_idx]} sentences\n")
        print(f"Q: {best_qa['instruction']}\n")
        print(f"A: {best_qa['response'][:500]}...")
        
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        # Generate recommendations
        if statistics.mean(response_words) < 60:
            print("⚠️  Average response length is low (<60 words)")
            print("   → Consider using improved prompts with more detailed requirements")
        
        if low_quality / len(qa_pairs) > 0.3:
            print("⚠️  High proportion of low-quality responses (>30%)")
            print("   → Increase quality filters or improve prompts")
        
        if len(question_starters) < 5:
            print("⚠️  Limited question diversity")
            print("   → Add more extraction strategies with varied question types")
        
        if len(source_counts) < 5 and len(qa_pairs) > 100:
            print("⚠️  Few source documents for dataset size")
            print("   → Add more source documents for better coverage")
        
        if statistics.mean(response_words) >= 70 and high_quality / len(qa_pairs) > 0.6:
            print("✅ Dataset quality is GOOD - ready for fine-tuning!")
            print("   → Responses are detailed and well-structured")
        
        if len(qa_pairs) >= 1000:
            print("✅ Dataset size is EXCELLENT (1000+ pairs)")
            print("   → Large enough for effective fine-tuning")
        elif len(qa_pairs) >= 500:
            print("✅ Dataset size is GOOD (500+ pairs)")
            print("   → Should work well for fine-tuning")
        else:
            print("⚠️  Dataset size is modest (<500 pairs)")
            print("   → Consider adding more source documents or improving extraction")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    analyzer = DatasetAnalyzer("output")
    analyzer.analyze_dataset()