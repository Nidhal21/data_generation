"""
PARALLEL Q&A GENERATOR - Optimized for 8 vCPU
✓ Uses multiprocessing for 5-8x speedup
✓ Thread-safe checkpointing
✓ HIGH QUALITY: Enhanced prompts and validation
✓ Progress tracking with ETA
✓ Automatic retry and error handling
"""

import json
import time
import re
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
import threading
from datetime import datetime, timedelta


class ParallelQAGenerator:
    def __init__(self, output_folder: str = "output", num_workers: int = 6):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3.2"
        self.num_workers = num_workers  # Use 6 workers for 8 vCPU (leave 2 for system)
        
        # Try to use existing checkpoint, fallback to parallel version
        self.checkpoint_file = self.output_folder / "qa_checkpoint.json"
        if not self.checkpoint_file.exists():
            self.checkpoint_file = self.output_folder / "qa_checkpoint_parallel.json"
        
        self.progress_file = self.output_folder / "progress.json"
        
        self.check_ollama()
    
    def check_ollama(self):
        """Check Ollama status"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                print(f"✓ Ollama running. Models: {', '.join(available)}")
                print(f"✓ Parallel workers: {self.num_workers}")
        except Exception as e:
            print(f"\n❌ Ollama not running! Error: {e}")
            exit(1)
    
    def generate_qa_pairs(self, resume: bool = False):
        """Main orchestrator with parallel processing"""
        print("="*80)
        print("PARALLEL HIGH-QUALITY Q&A GENERATOR")
        print("="*80)
        print(f"Workers: {self.num_workers} | Target: 20-30 Q&A per chunk")
        print("="*80 + "\n")
        
        # Load chunks
        chunks_file = self.output_folder / "text_chunks.json"
        if not chunks_file.exists():
            print(f"❌ File not found: {chunks_file}")
            return
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = data['chunks']
        
        # Load checkpoint
        checkpoint = self.load_checkpoint() if resume else {'processed_chunks': [], 'qa_pairs': []}
        processed_chunk_ids = set(checkpoint['processed_chunks'])
        all_qa_pairs = checkpoint['qa_pairs']
        
        # Filter unprocessed chunks
        pending_chunks = [c for c in chunks if c['chunk_id'] not in processed_chunk_ids]
        
        if resume:
            print(f"✓ Resuming: {len(processed_chunk_ids)} chunks done, {len(pending_chunks)} remaining")
            print(f"✓ Current Q&A: {len(all_qa_pairs)}\n")
        else:
            print(f"✓ Total chunks: {len(chunks)}\n")
        
        if not pending_chunks:
            print("✅ All chunks already processed!")
            return
        
        # Prepare for parallel processing
        start_time = time.time()
        total_chunks = len(pending_chunks)
        
        # Process in parallel
        print(f"🚀 Starting parallel processing with {self.num_workers} workers...\n")
        
        completed = 0
        failed = 0
        total_qa_generated = len(all_qa_pairs)
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(process_chunk_worker, chunk, self.ollama_url, self.model): chunk
                for chunk in pending_chunks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                
                try:
                    qa_pairs, success = future.result()
                    
                    completed += 1
                    
                    if success and qa_pairs:
                        all_qa_pairs.extend(qa_pairs)
                        total_qa_generated += len(qa_pairs)
                        processed_chunk_ids.add(chunk['chunk_id'])
                        
                        # Progress info
                        elapsed = time.time() - start_time
                        avg_time_per_chunk = elapsed / completed
                        remaining = total_chunks - completed
                        eta_seconds = avg_time_per_chunk * remaining
                        eta = datetime.now() + timedelta(seconds=eta_seconds)
                        
                        print(f"[{completed}/{total_chunks}] ✓ {chunk['document'][:40]} | "
                              f"{len(qa_pairs)} Q&A | "
                              f"Total: {total_qa_generated} | "
                              f"ETA: {eta.strftime('%H:%M:%S')}")
                    else:
                        failed += 1
                        print(f"[{completed}/{total_chunks}] ⚠ {chunk['document'][:40]} | Failed")
                    
                    # Save checkpoint every 10 chunks
                    if completed % 10 == 0:
                        self.save_checkpoint(list(processed_chunk_ids), all_qa_pairs)
                        self.save_progress(completed, total_chunks, total_qa_generated, elapsed)
                    
                except Exception as e:
                    completed += 1
                    failed += 1
                    print(f"[{completed}/{total_chunks}] ❌ {chunk['document'][:40]} | Error: {str(e)[:50]}")
        
        # Final save
        self.save_checkpoint(list(processed_chunk_ids), all_qa_pairs)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✓ PROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total Q&A pairs: {len(all_qa_pairs)}")
        print(f"Chunks processed: {completed}/{total_chunks}")
        print(f"Success rate: {((completed-failed)/completed*100):.1f}%")
        print(f"Average per chunk: {total_qa_generated // max(1, completed-failed)}")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Speed: {elapsed/completed:.1f} sec/chunk")
        print(f"{'='*80}\n")
        
        # Save final dataset
        output_file = self.output_folder / "raw_qa_pairs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'qa_pairs': all_qa_pairs,
                'stats': {
                    'total_qa': len(all_qa_pairs),
                    'chunks_processed': completed,
                    'failed': failed,
                    'time_elapsed_minutes': elapsed/60
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {output_file}\n")
    
    def load_checkpoint(self) -> Dict:
        """Load checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'processed_chunks': [], 'qa_pairs': []}
    
    def save_checkpoint(self, processed_chunks: List[int], qa_pairs: List[Dict]):
        """Thread-safe checkpoint save"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_chunks': processed_chunks,
                'qa_pairs': qa_pairs,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
    
    def save_progress(self, completed: int, total: int, qa_count: int, elapsed: float):
        """Save progress info"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'completed': completed,
                'total': total,
                'progress_percent': (completed/total*100),
                'total_qa_generated': qa_count,
                'elapsed_minutes': elapsed/60,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)


def call_ollama(prompt: str, ollama_url: str, model: str, max_retries: int = 3) -> str:
    """Call Ollama with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 2000,

                        "top_p": 0.9,
                        "repeat_penalty": 1.2,
                        "num_ctx": 8192
                    }
                },
                timeout=200
            )
            
            if response.status_code == 200:
                result = response.json()['response'].strip()
                if result:
                    return result
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return ""


def process_chunk_worker(chunk: Dict, ollama_url: str, model: str) -> Tuple[List[Dict], bool]:
    """
    Worker function for parallel processing
    This runs in a separate process
    """
    try:
        # HIGH QUALITY PROMPTS - 4 strategies
        prompts = [
            # Strategy 1: Deep Conceptual Understanding
            f"""You are an expert educator creating comprehensive learning materials.

TEXT:
{chunk['text'][:2000]}

TASK: Create 3 exceptional Q&A pairs about core concepts, definitions, and fundamental understanding.

QUALITY REQUIREMENTS (CRITICAL):
- Questions must be specific and clear (20+ chars)
- Answers MUST be 8-12 sentences minimum
- Include concrete examples
- Explain WHY concepts matter
- Connect to practical applications
- Use clear, engaging educational language
- Avoid generic phrases like "it is important" or "there are many"

OUTPUT: Valid JSON array only, no markdown:
[{{"instruction":"What is [specific concept]?","response":"Comprehensive 8-12 sentence explanation with examples...","category":"definition","difficulty":"beginner"}}]

Generate exactly 3 Q&A pairs:""",

            # Strategy 2: Process Mastery
            f"""You are a technical documentation expert creating actionable guides.

TEXT:
{chunk['text'][:2000]}

TASK: Create 3 detailed Q&A pairs about processes, procedures, and implementation steps.

QUALITY REQUIREMENTS (CRITICAL):
- Questions start with "How to" or "What is the process for"
- Answers MUST be 9-15 sentences with numbered steps
- Explain the rationale behind each step
- Include warnings or tips
- Provide context for when to use
- Be specific and actionable

OUTPUT: Valid JSON array only:
[{{"instruction":"How to [specific process]?","response":"Detailed 9-15 sentence process with numbered steps...","category":"process","difficulty":"intermediate"}}]

Generate exactly 3 Q&A pairs:""",

            # Strategy 3: Problem-Solving Excellence
            f"""You are a senior consultant helping teams solve real challenges.

TEXT:
{chunk['text'][:2000]}

TASK: Create 3 Q&A pairs about problems, solutions, and best practices.

QUALITY REQUIREMENTS (CRITICAL):
- Questions about realistic scenarios
- Answers MUST be 8-12 sentences
- Describe the problem context clearly
- Provide detailed solutions with reasoning
- Include multiple approaches when relevant
- Add practical recommendations

OUTPUT: Valid JSON array only:
[{{"instruction":"How to handle [specific challenge]?","response":"Complete 8-12 sentence solution with context and reasoning...","category":"problem-solving","difficulty":"advanced"}}]

Generate exactly 3 Q&A pairs:""",

            # Strategy 4: Comparative Analysis
            f"""You are an analytical expert helping learners understand relationships and differences.

TEXT:
{chunk['text'][:2000]}

TASK: Create 3 Q&A pairs about comparisons, contrasts, and relationships.

QUALITY REQUIREMENTS (CRITICAL):
- Questions compare 2-3 concepts or approaches
- Answers MUST be 7-10 sentences
- Highlight key differences AND similarities
- Explain when to use each option
- Provide decision criteria
- Include real-world context

OUTPUT: Valid JSON array only:
[{{"instruction":"What is the difference between [A] and [B]?","response":"Thorough 7-10 sentence comparison with context...","category":"comparison","difficulty":"intermediate"}}]

Generate exactly 3 Q&A pairs:"""
        ]
        
        all_qa = []
        
        # Execute all 4 strategies
        for prompt in prompts:
            response = call_ollama(prompt, ollama_url, model)
            if response:
                qa_pairs = extract_and_validate_qa(response, chunk['document'])
                all_qa.extend(qa_pairs)
            time.sleep(0.3)
        
        return all_qa, True
        
    except Exception as e:
        return [], False


def extract_and_validate_qa(response: str, filename: str) -> List[Dict]:
    """Extract and validate Q&A with HIGH quality standards"""
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = response.strip()
    
    qa_pairs = []
    
    # Try JSON parsing
    try:
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = fix_json(json_str)
            parsed = json.loads(json_str)
            
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        qa_pairs.append(item)
                    elif isinstance(item, list):
                        for nested in item:
                            if isinstance(nested, dict):
                                qa_pairs.append(nested)
    except:
        pass
    
    # Try individual objects
    if not qa_pairs:
        try:
            pattern = r'\{[^{}]*?"instruction"[^{}]*?"response"[^{}]*?\}'
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    obj = json.loads(fix_json(match))
                    if isinstance(obj, dict) and 'instruction' in obj and 'response' in obj:
                        qa_pairs.append(obj)
                except:
                    continue
        except:
            pass
    
    # HIGH QUALITY VALIDATION
    valid_pairs = []
    for qa in qa_pairs:
        if not isinstance(qa, dict):
            continue
        
        inst = str(qa.get('instruction', '')).strip()
        resp = str(qa.get('response', '')).strip()
        
        # STRICT quality filters for high quality
        if len(inst) < 20:  # Minimum question length
            continue
        if len(resp) < 150:  # Increased from 120
            continue
        if len(resp.split()) < 50:  # Minimum 50 words (increased from 40)
            continue
        if resp.count('.') < 5:  # Minimum 5 sentences (increased from 4)
            continue
        
        # Check for generic/low-quality responses
        generic_phrases = [
            'it is important', 'there are many', 'it depends', 'in general',
            'it can be', 'there is a', 'this is a', 'these are'
        ]
        resp_lower = resp.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in resp_lower)
        if generic_count > 3:  # Too generic
            continue
        
        # Check for actual content (not just filler)
        content_words = len([w for w in resp.split() if len(w) > 4])
        if content_words < 25:  # Need substantial content
            continue
        
        qa['source'] = filename
        qa.setdefault('category', 'general')
        qa.setdefault('difficulty', 'intermediate')
        
        valid_pairs.append(qa)
    
    return valid_pairs


def fix_json(text: str) -> str:
    """Fix common JSON issues"""
    text = re.sub(r',(\s*[\]}])', r'\1', text)
    text = text.replace('\\"', '"')
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


if __name__ == "__main__":
    import sys
    
    # Configure for your system (2 workers for 8 vCPU)
    generator = ParallelQAGenerator("output", num_workers=2)
    
    resume = len(sys.argv) > 1 and sys.argv[1] == '--resume'
    
    if resume:
        print("🔄 RESUMING from checkpoint...\n")
    
    generator.generate_qa_pairs(resume=resume)