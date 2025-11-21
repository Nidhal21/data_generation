"""
MASTER RUNNER - Execute all steps or individual steps
Usage:
  python run_all.py                 # Run all steps
  python run_all.py 1               # Run only step 1 (extraction)
  python run_all.py 2               # Run only step 2 (chunking)
  python run_all.py 3               # Run only step 3 (Q&A generation)
  python run_all.py 3 --resume      # Resume step 3 from checkpoint
  python run_all.py 4               # Run only step 4 (finalization)
  python run_all.py 2-4             # Run steps 2 through 4
"""

import sys
import subprocess
from pathlib import Path


class PipelineRunner:
    def __init__(self):
        self.steps = {
            1: ("1_document_processor.py", "Document Extraction & Translation"),
            2: ("2_text_chunker.py", "Text Chunking"),
            3: ("3_qa_generator.py", "Q&A Generation"),
            4: ("4_finalizer.py", "Dataset Finalization")
        }
    
    def run_step(self, step_num: int, args: list = None):
        """Run a single step"""
        if step_num not in self.steps:
            print(f"❌ Invalid step: {step_num}")
            return False
        
        script, description = self.steps[step_num]
        
        if not Path(script).exists():
            print(f"❌ Script not found: {script}")
            return False
        
        print(f"\n{'='*80}")
        print(f"RUNNING STEP {step_num}: {description}")
        print(f"{'='*80}\n")
        
        cmd = [sys.executable, script]
        if args:
            cmd.extend(args)
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n❌ Step {step_num} failed with error code {result.returncode}")
            return False
        
        print(f"\n✅ Step {step_num} completed successfully!")
        return True
    
    def run_all(self):
        """Run all steps in sequence"""
        print("\n" + "="*80)
        print("MISTRAL FINE-TUNING DATASET PIPELINE")
        print("="*80)
        print("This will run all 4 steps:")
        print("  1. Extract & translate documents")
        print("  2. Create text chunks")
        print("  3. Generate Q&A pairs")
        print("  4. Finalize dataset")
        print("="*80 + "\n")
        
        for step_num in range(1, 5):
            if not self.run_step(step_num):
                print(f"\n❌ Pipeline stopped at step {step_num}")
                return False
        
        print("\n" + "="*80)
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nYour dataset is ready in the 'output' folder:")
        print("  - mistral_dataset_*pairs.jsonl  <- Use this for Mistral fine-tuning")
        print("  - mistral_dataset_detailed.json <- Full dataset with metadata")
        print("="*80 + "\n")
        return True
    
    def run_range(self, start: int, end: int):
        """Run a range of steps"""
        for step_num in range(start, end + 1):
            if not self.run_step(step_num):
                return False
        return True


def print_usage():
    """Print usage instructions"""
    print(__doc__)


def main():
    runner = PipelineRunner()
    
    # No arguments - run all
    if len(sys.argv) == 1:
        runner.run_all()
        return
    
    arg = sys.argv[1]
    
    # Help
    if arg in ['-h', '--help', 'help']:
        print_usage()
        return
    
    # Range (e.g., "2-4")
    if '-' in arg:
        try:
            start, end = map(int, arg.split('-'))
            runner.run_range(start, end)
        except:
            print(f"❌ Invalid range: {arg}")
            print_usage()
        return
    
    # Single step
    try:
        step_num = int(arg)
        
        # Check for --resume flag
        extra_args = []
        if len(sys.argv) > 2 and sys.argv[2] == '--resume':
            extra_args.append('--resume')
        
        runner.run_step(step_num, extra_args)
    except ValueError:
        print(f"❌ Invalid step number: {arg}")
        print_usage()


if __name__ == "__main__":
    main()