"""
PREMIUM Data Cleaning & Preparation Script
Maximum Quality Configuration for Fine-tuning
Enhanced with advanced quality controls
"""

import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import hashlib
from tqdm import tqdm
import unicodedata

# ============================================================================
# PREMIUM CONFIGURATION - MAXIMUM QUALITY
# ============================================================================

class PremiumCleaningConfig:
    # Input/Output files
    FR_INPUT = "final_data_fr.jsonl"
    EN_INPUT = "final_data_eng.jsonl"
    FR_OUTPUT = "final_data_fr_premium.jsonl"
    EN_OUTPUT = "final_data_eng_premium.jsonl"
    
    # === STRICT LENGTH CONSTRAINTS (Premium Quality) ===
    MIN_QUESTION_LENGTH = 15      # Increased from 10
    MAX_QUESTION_LENGTH = 800     # Reduced from 1000
    MIN_ANSWER_LENGTH = 50        # Increased from 20 - ensure detailed answers
    MAX_ANSWER_LENGTH = 2500      # Reduced from 3000 - prevent truncation
    
    # === WORD COUNT (Premium Quality) ===
    MIN_QUESTION_WORDS = 5        # Increased from 3
    MAX_QUESTION_WORDS = 150      # Prevent overly long questions
    MIN_ANSWER_WORDS = 20         # Increased from 10 - ensure quality
    MAX_ANSWER_WORDS = 500        # Prevent rambling
    
    # === DUPLICATE DETECTION (Aggressive) ===
    REMOVE_DUPLICATES = True
    REMOVE_NEAR_DUPLICATES = True
    SIMILARITY_THRESHOLD = 0.85   # More aggressive (was 0.90)
    USE_FUZZY_MATCHING = True     # New: Advanced fuzzy matching
    
    # === TEXT QUALITY (Premium) ===
    FIX_FORMATTING = True
    NORMALIZE_WHITESPACE = True
    NORMALIZE_UNICODE = True      # New: Fix encoding issues
    FIX_PUNCTUATION = True
    FIX_CAPITALIZATION = True     # New: Fix inconsistent caps
    REMOVE_URLS = True            # New: Remove URLs
    REMOVE_EMAIL_ADDRESSES = True # New: Remove emails
    REMOVE_SPECIAL_CHARS = False
    
    # === CONTENT QUALITY (Premium) ===
    MIN_SENTENCE_COUNT_ANSWER = 2  # New: Ensure multi-sentence answers
    REMOVE_INCOMPLETE = True
    REMOVE_LOW_INFORMATION = True  # New: Remove generic/vague answers
    CHECK_LANGUAGE_CONSISTENCY = True
    ENSURE_PM_RELEVANCE = True    # New: Ensure PM domain relevance
    
    # === ADVANCED FILTERS (New) ===
    REMOVE_QUESTIONS_WITH_PLACEHOLDERS = True  # Remove [...], XXX, etc.
    REMOVE_REPETITIVE_TEXT = True             # Remove repetitive patterns
    CHECK_GRAMMAR_BASIC = True                # Basic grammar validation
    BALANCE_LANGUAGES = True                  # Ensure 50/50 FR/EN
    
    # === PM DOMAIN KEYWORDS (For relevance check) ===
    PM_KEYWORDS_EN = {
        'scrum', 'agile', 'sprint', 'backlog', 'project', 'management',
        'stakeholder', 'requirement', 'risk', 'scope', 'schedule', 'budget',
        'team', 'deliverable', 'milestone', 'kanban', 'waterfall', 'iteration',
        'planning', 'estimation', 'velocity', 'burndown', 'retrospective',
        'product owner', 'scrum master', 'pmp', 'pmbok', 'gantt', 'wbs'
    }
    
    PM_KEYWORDS_FR = {
        'scrum', 'agile', 'sprint', 'backlog', 'projet', 'gestion',
        'partie prenante', 'exigence', 'risque', 'portée', 'calendrier',
        'équipe', 'livrable', 'jalon', 'kanban', 'itération', 'planification',
        'estimation', 'vélocité', 'rétrospective', 'product owner',
        'scrum master', 'cascade', 'méthodologie'
    }
    
    # === QUALITY SCORES ===
    MIN_QUALITY_SCORE = 7.0  # Out of 10

# ============================================================================
# ADVANCED TEXT CLEANING
# ============================================================================

class PremiumTextCleaner:
    """Premium text cleaning with advanced features"""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters"""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        # Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # BOM
        return text
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs"""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_email_addresses(text: str) -> str:
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def remove_placeholders(text: str) -> bool:
        """Check if text contains placeholders"""
        placeholders = [
            r'\[.*?\]',           # [placeholder]
            r'\{.*?\}',           # {placeholder}
            r'XXX+',              # XXX
            r'TODO',              # TODO
            r'\.\.\.',            # ...
            r'<.*?>',             # <placeholder>
        ]
        for pattern in placeholders:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def fix_capitalization(text: str, language: str) -> str:
        """Fix capitalization issues"""
        if not text:
            return text
        
        # Capitalize first letter
        if text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Fix common PM terms
        pm_terms = {
            'scrum': 'Scrum',
            'agile': 'Agile',
            'kanban': 'Kanban',
            'product owner': 'Product Owner',
            'scrum master': 'Scrum Master',
        }
        
        for incorrect, correct in pm_terms.items():
            # Only replace at word boundaries
            pattern = r'\b' + re.escape(incorrect) + r'\b'
            text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Advanced whitespace normalization"""
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Replace multiple spaces
        text = re.sub(r' +', ' ', text)
        # Fix space before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Fix space after punctuation
        text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', text)
        # Remove leading/trailing
        text = text.strip()
        return text
    
    @staticmethod
    def fix_punctuation(text: str) -> str:
        """Advanced punctuation fixes"""
        if not text:
            return text
        
        # Fix multiple punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r',{2,}', ',', text)
        
        # Fix space before comma/period
        text = re.sub(r'\s+([.,])', r'\1', text)
        
        # Ensure sentence ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Fix multiple spaces after punctuation
        text = re.sub(r'([.!?,;:])\s+', r'\1 ', text)
        
        return text
    
    @staticmethod
    def remove_repetitive_patterns(text: str) -> bool:
        """Detect repetitive patterns"""
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for repeated 3-word sequences
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        
        # If any trigram appears more than twice, it's repetitive
        for count in trigram_counts.values():
            if count > 2:
                return True
        
        return False
    
    @staticmethod
    def clean_text(text: str, language: str, config) -> str:
        """Premium text cleaning"""
        if not text:
            return ""
        
        # Unicode normalization
        if config.NORMALIZE_UNICODE:
            text = PremiumTextCleaner.normalize_unicode(text)
        
        # Remove URLs and emails
        if config.REMOVE_URLS:
            text = PremiumTextCleaner.remove_urls(text)
        if config.REMOVE_EMAIL_ADDRESSES:
            text = PremiumTextCleaner.remove_email_addresses(text)
        
        # Basic cleaning
        if config.NORMALIZE_WHITESPACE:
            text = PremiumTextCleaner.normalize_whitespace(text)
        if config.FIX_PUNCTUATION:
            text = PremiumTextCleaner.fix_punctuation(text)
        
        # Fix capitalization
        if config.FIX_CAPITALIZATION:
            text = PremiumTextCleaner.fix_capitalization(text, language)
        
        # Language-specific
        if language == 'fr':
            # Fix French quotes
            text = re.sub(r'«\s*', '« ', text)
            text = re.sub(r'\s*»', ' »', text)
            # Fix apostrophes
            text = text.replace("' ", "'")
        else:
            # Standardize English quotes
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()

# ============================================================================
# PREMIUM QUALITY VALIDATORS
# ============================================================================

class PremiumQualityValidator:
    """Premium quality validation with scoring"""
    
    @staticmethod
    def calculate_quality_score(question: str, answer: str, language: str, config) -> float:
        """Calculate quality score (0-10)"""
        score = 10.0
        
        # Length penalties
        q_words = len(question.split())
        a_words = len(answer.split())
        
        if q_words < 5:
            score -= 2.0
        if a_words < 20:
            score -= 2.0
        if a_words > 500:
            score -= 1.0
        
        # Sentence count in answer
        sentences = re.split(r'[.!?]+', answer)
        sentence_count = len([s for s in sentences if s.strip()])
        if sentence_count < 2:
            score -= 1.5
        
        # Check for PM relevance
        keywords = config.PM_KEYWORDS_FR if language == 'fr' else config.PM_KEYWORDS_EN
        text_lower = (question + ' ' + answer).lower()
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        
        if keyword_count < 2:
            score -= 2.0  # Not very relevant to PM
        elif keyword_count < 1:
            score -= 3.0  # Not relevant at all
        
        # Check for placeholders
        if PremiumTextCleaner.remove_placeholders(question) or \
           PremiumTextCleaner.remove_placeholders(answer):
            score -= 3.0
        
        # Check for repetitive patterns
        if PremiumTextCleaner.remove_repetitive_patterns(answer):
            score -= 2.0
        
        # Punctuation quality
        if not answer.strip()[-1] in '.!?':
            score -= 1.0
        
        return max(0.0, score)
    
    @staticmethod
    def is_low_information(answer: str) -> bool:
        """Check if answer is generic/low information"""
        low_info_patterns = [
            r'^(yes|no|maybe|it depends)\.?$',
            r'^(oui|non|peut-être|ça dépend)\.?$',
            r'^.{1,30}$',  # Very short answers
            r'^(i don\'t know|je ne sais pas)',
        ]
        
        answer_lower = answer.lower().strip()
        for pattern in low_info_patterns:
            if re.match(pattern, answer_lower, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def check_pm_relevance(text: str, language: str, config) -> bool:
        """Check if text is relevant to project management"""
        keywords = config.PM_KEYWORDS_FR if language == 'fr' else config.PM_KEYWORDS_EN
        text_lower = text.lower()
        
        # Count keyword occurrences
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        
        # At least 1 PM keyword required
        return keyword_count >= 1
    
    @staticmethod
    def has_good_structure(answer: str, config) -> bool:
        """Check if answer has good structure"""
        # Count sentences
        sentences = re.split(r'[.!?]+', answer)
        sentence_count = len([s for s in sentences if len(s.strip()) > 10])
        
        return sentence_count >= config.MIN_SENTENCE_COUNT_ANSWER
    
    @staticmethod
    def check_grammar_basic(text: str, language: str) -> bool:
        """Basic grammar checks"""
        # Check for repeated words
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i+1] and len(words[i]) > 3:
                return False
        
        # Check for proper sentence structure (capital start)
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                return False
        
        return True

# ============================================================================
# ADVANCED DUPLICATE DETECTION
# ============================================================================

class AdvancedDuplicateDetector:
    """Advanced duplicate detection with fuzzy matching"""
    
    @staticmethod
    def compute_fuzzy_hash(text: str) -> str:
        """Compute fuzzy hash (ignores minor variations)"""
        # Normalize heavily
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return hashlib.md5(text.encode()).hexdigest()
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Compute Jaccard similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def find_all_duplicates(questions: List[str], threshold: float, use_fuzzy: bool) -> Set[int]:
        """Find all duplicate indices"""
        n = len(questions)
        to_remove = set()
        
        # Exact duplicates using hash
        seen_hashes = {}
        for i, question in enumerate(questions):
            if use_fuzzy:
                hash_val = AdvancedDuplicateDetector.compute_fuzzy_hash(question)
            else:
                hash_val = hashlib.md5(question.lower().encode()).hexdigest()
            
            if hash_val in seen_hashes:
                to_remove.add(i)
            else:
                seen_hashes[hash_val] = i
        
        print(f"  Exact/fuzzy duplicates: {len(to_remove)}")
        
        # Near duplicates using similarity
        remaining_indices = [i for i in range(n) if i not in to_remove]
        print(f"  Checking {len(remaining_indices)} for near-duplicates...")
        
        for idx, i in enumerate(tqdm(remaining_indices, desc="  Near-duplicate scan")):
            if i in to_remove:
                continue
            
            for j in remaining_indices[idx+1:]:
                if j in to_remove:
                    continue
                
                similarity = AdvancedDuplicateDetector.jaccard_similarity(
                    questions[i], questions[j]
                )
                
                if similarity >= threshold:
                    to_remove.add(j)
        
        return to_remove

# ============================================================================
# PREMIUM DATA CLEANER
# ============================================================================

class PremiumDataCleaner:
    """Premium data cleaning with maximum quality"""
    
    def __init__(self, config: PremiumCleaningConfig):
        self.config = config
        self.stats = defaultdict(int)
        self.quality_scores = []
    
    def load_data(self, filepath: str) -> List[Dict]:
        """Load JSONL data"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data
    
    def process_dataset(self, input_file: str, output_file: str, language: str):
        """Process dataset with premium quality"""
        print("\n" + "="*70)
        print(f"✨ PREMIUM CLEANING - {language.upper()} DATASET")
        print("="*70)
        
        # Load
        print(f"\n📂 Loading {input_file}...")
        data = self.load_data(input_file)
        self.stats['original'] = len(data)
        print(f"  Original: {len(data):,} items")
        
        # Step 1: Structure validation
        print("\n1️⃣  Structure Validation...")
        data = self._filter_structure(data)
        
        # Step 2: Clean text
        print("\n2️⃣  Premium Text Cleaning...")
        data = self._clean_all_text(data, language)
        
        # Step 3: Length filters
        print("\n3️⃣  Length Validation...")
        data = self._filter_length(data)
        
        # Step 4: Quality filters
        print("\n4️⃣  Quality Validation...")
        data = self._filter_quality(data, language)
        
        # Step 5: Advanced filters
        print("\n5️⃣  Advanced Filters...")
        data = self._advanced_filters(data, language)
        
        # Step 6: Quality scoring
        print("\n6️⃣  Quality Scoring...")
        data = self._score_and_filter(data, language)
        
        # Step 7: Duplicate removal
        print("\n7️⃣  Duplicate Removal...")
        data = self._remove_duplicates(data)
        
        self.stats['final'] = len(data)
        
        # Save
        print(f"\n💾 Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  ✓ Saved: {len(data):,} premium items")
        
        # Statistics
        self._print_stats(language)
        
        return data
    
    def _filter_structure(self, data: List[Dict]) -> List[Dict]:
        """Filter by structure"""
        valid = []
        for item in data:
            try:
                msgs = item['messages']
                if len(msgs) == 2 and \
                   msgs[0]['role'] == 'user' and \
                   msgs[1]['role'] == 'assistant' and \
                   msgs[0]['content'] and msgs[1]['content']:
                    valid.append(item)
                else:
                    self.stats['removed_structure'] += 1
            except:
                self.stats['removed_structure'] += 1
        
        print(f"  ✓ Valid structure: {len(valid):,}/{len(data):,}")
        return valid
    
    def _clean_all_text(self, data: List[Dict], language: str) -> List[Dict]:
        """Clean all text"""
        cleaned = []
        for item in tqdm(data, desc="  Cleaning"):
            question = item['messages'][0]['content']
            answer = item['messages'][1]['content']
            
            question = PremiumTextCleaner.clean_text(question, language, self.config)
            answer = PremiumTextCleaner.clean_text(answer, language, self.config)
            
            cleaned.append({
                'messages': [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ]
            })
        
        return cleaned
    
    def _filter_length(self, data: List[Dict]) -> List[Dict]:
        """Filter by length"""
        valid = []
        for item in data:
            q = item['messages'][0]['content']
            a = item['messages'][1]['content']
            
            q_len, a_len = len(q), len(a)
            q_words, a_words = len(q.split()), len(a.split())
            
            # Character length
            if not (self.config.MIN_QUESTION_LENGTH <= q_len <= self.config.MAX_QUESTION_LENGTH):
                self.stats['removed_question_length'] += 1
                continue
            if not (self.config.MIN_ANSWER_LENGTH <= a_len <= self.config.MAX_ANSWER_LENGTH):
                self.stats['removed_answer_length'] += 1
                continue
            
            # Word count
            if not (self.config.MIN_QUESTION_WORDS <= q_words <= self.config.MAX_QUESTION_WORDS):
                self.stats['removed_question_words'] += 1
                continue
            if not (self.config.MIN_ANSWER_WORDS <= a_words <= self.config.MAX_ANSWER_WORDS):
                self.stats['removed_answer_words'] += 1
                continue
            
            valid.append(item)
        
        print(f"  ✓ Valid length: {len(valid):,}/{len(data):,}")
        return valid
    
    def _filter_quality(self, data: List[Dict], language: str) -> List[Dict]:
        """Filter by quality"""
        valid = []
        for item in data:
            q = item['messages'][0]['content']
            a = item['messages'][1]['content']
            
            # Check completeness
            if self.config.REMOVE_INCOMPLETE:
                if not a.strip()[-1] in '.!?':
                    self.stats['removed_incomplete'] += 1
                    continue
            
            # Check low information
            if self.config.REMOVE_LOW_INFORMATION:
                if PremiumQualityValidator.is_low_information(a):
                    self.stats['removed_low_info'] += 1
                    continue
            
            # Check structure
            if not PremiumQualityValidator.has_good_structure(a, self.config):
                self.stats['removed_structure_bad'] += 1
                continue
            
            # Check grammar
            if self.config.CHECK_GRAMMAR_BASIC:
                if not PremiumQualityValidator.check_grammar_basic(q, language):
                    self.stats['removed_grammar'] += 1
                    continue
            
            # Check PM relevance
            if self.config.ENSURE_PM_RELEVANCE:
                if not PremiumQualityValidator.check_pm_relevance(
                    q + ' ' + a, language, self.config
                ):
                    self.stats['removed_not_pm'] += 1
                    continue
            
            valid.append(item)
        
        print(f"  ✓ Valid quality: {len(valid):,}/{len(data):,}")
        return valid
    
    def _advanced_filters(self, data: List[Dict], language: str) -> List[Dict]:
        """Apply advanced filters"""
        valid = []
        for item in data:
            q = item['messages'][0]['content']
            a = item['messages'][1]['content']
            
            # Check placeholders
            if self.config.REMOVE_QUESTIONS_WITH_PLACEHOLDERS:
                if PremiumTextCleaner.remove_placeholders(q) or \
                   PremiumTextCleaner.remove_placeholders(a):
                    self.stats['removed_placeholders'] += 1
                    continue
            
            # Check repetitive
            if self.config.REMOVE_REPETITIVE_TEXT:
                if PremiumTextCleaner.remove_repetitive_patterns(a):
                    self.stats['removed_repetitive'] += 1
                    continue
            
            valid.append(item)
        
        print(f"  ✓ Passed advanced: {len(valid):,}/{len(data):,}")
        return valid
    
    def _score_and_filter(self, data: List[Dict], language: str) -> List[Dict]:
        """Score and filter by quality score"""
        scored_data = []
        
        for item in tqdm(data, desc="  Scoring"):
            q = item['messages'][0]['content']
            a = item['messages'][1]['content']
            
            score = PremiumQualityValidator.calculate_quality_score(
                q, a, language, self.config
            )
            
            self.quality_scores.append(score)
            
            if score >= self.config.MIN_QUALITY_SCORE:
                scored_data.append(item)
            else:
                self.stats['removed_low_score'] += 1
        
        avg_score = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        print(f"  ✓ High quality: {len(scored_data):,}/{len(data):,}")
        print(f"  Average score: {avg_score:.2f}/10")
        
        return scored_data
    
    def _remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        questions = [item['messages'][0]['content'] for item in data]
        
        to_remove = AdvancedDuplicateDetector.find_all_duplicates(
            questions,
            self.config.SIMILARITY_THRESHOLD,
            self.config.USE_FUZZY_MATCHING
        )
        
        self.stats['removed_duplicates'] = len(to_remove)
        
        valid_data = [item for i, item in enumerate(data) if i not in to_remove]
        
        print(f"  ✓ Unique items: {len(valid_data):,}/{len(data):,}")
        
        return valid_data
    
    def _print_stats(self, language: str):
        """Print statistics"""
        print("\n" + "="*70)
        print(f"📊 {language.upper()} PREMIUM CLEANING STATISTICS")
        print("="*70)
        
        print(f"\nOriginal count:     {self.stats['original']:,}")
        print(f"Final count:        {self.stats['final']:,}")
        
        removed = self.stats['original'] - self.stats['final']
        pct = (removed / self.stats['original'] * 100) if self.stats['original'] > 0 else 0
        print(f"Removed:            {removed:,} ({pct:.1f}%)")
        print(f"Retention:          {(100-pct):.1f}%")
        
        if self.quality_scores:
            print(f"\n✨ Quality Metrics:")
            print(f"  Average score:    {sum(self.quality_scores)/len(self.quality_scores):.2f}/10")
            print(f"  Min score:        {min(self.quality_scores):.2f}/10")
            print(f"  Max score:        {max(self.quality_scores):.2f}/10")
        
        print(f"\n📋 Removal Breakdown:")
        for key, value in sorted(self.stats.items()):
            if key.startswith('removed_') and value > 0:
                reason = key.replace('removed_', '').replace('_', ' ').title()
                print(f"  {reason:.<35} {value:,}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("="*70)
    print("✨ PREMIUM DATA CLEANING - MAXIMUM QUALITY")
    print("="*70)
    
    config = PremiumCleaningConfig()
    
    # French dataset
    cleaner_fr = PremiumDataCleaner(config)
    fr_data = cleaner_fr.process_dataset(
        config.FR_INPUT,
        config.FR_OUTPUT,
        'fr'
    )
    
    # English dataset
    cleaner_en = PremiumDataCleaner(config)
    en_data = cleaner_en.process_dataset(
        config.EN_INPUT,
        config.EN_OUTPUT,
        'en'
    )
    
    # Balance languages if enabled
    if config.BALANCE_LANGUAGES:
        print("\n" + "="*70)
        print("⚖️  BALANCING LANGUAGES")
        print("="*70)
        
        min_size = min(len(fr_data), len(en_data))
        print(f"\nBalancing to {min_size:,} examples per language")
        
        # Keep first N items (already high quality)
        fr_data = fr_data[:min_size]
        en_data = en_data[:min_size]
        
        # Re-save
        with open(config.FR_OUTPUT, 'w', encoding='utf-8') as f:
            for item in fr_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(config.EN_OUTPUT, 'w', encoding='utf-8') as f:
            for item in en_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PREMIUM CLEANING COMPLETE")
    print("="*70)
    
    print(f"\n✨ Premium Quality Dataset Created:")
    print(f"  French:   {len(fr_data):,} examples")
    print(f"  English:  {len(en_data):,} examples")
    print(f"  Total:    {len(fr_data) + len(en_data):,} examples")
    print(f"  Balance:  50/50 ✅" if len(fr_data) == len(en_data) else f"  Balance:  {len(fr_data)}/{len(en_data)}")
    
    print(f"\n📁 Output Files:")
    print(f"  {config.FR_OUTPUT}")
    print(f"  {config.EN_OUTPUT}")
    
    print(f"\n🏆 Quality Assurance:")
    print(f"  ✅ All duplicates removed")
    print(f"  ✅ PM domain relevance verified")
    print(f"  ✅ Minimum quality score: {config.MIN_QUALITY_SCORE}/10")
    print(f"  ✅ Proper length constraints")
    print(f"  ✅ Grammar validated")
    print(f"  ✅ Formatting standardized")
    
    print(f"\n🚀 Ready for Premium Fine-tuning!")
    print(f"   Update your training script:")
    print(f"   FR_DATA_PATH = '{config.FR_OUTPUT}'")
    print(f"   EN_DATA_PATH = '{config.EN_OUTPUT}'")
    
    # Save metadata
    metadata = {
        'total_examples': len(fr_data) + len(en_data),
        'french_examples': len(fr_data),
        'english_examples': len(en_data),
        'quality_threshold': config.MIN_QUALITY_SCORE,
        'similarity_threshold': config.SIMILARITY_THRESHOLD,
        'min_answer_words': config.MIN_ANSWER_WORDS,
        'avg_quality_score_fr': sum(cleaner_fr.quality_scores) / len(cleaner_fr.quality_scores) if cleaner_fr.quality_scores else 0,
        'avg_quality_score_en': sum(cleaner_en.quality_scores) / len(cleaner_en.quality_scores) if cleaner_en.quality_scores else 0,
    }
    
    with open('premium_cleaning_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Metadata saved to: premium_cleaning_metadata.json")

if __name__ == "__main__":
    main()