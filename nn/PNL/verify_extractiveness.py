import re
from collections import Counter
from nltk.util import ngrams

def get_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def calculate_novel_ngrams(source_text, target_text, n):
    source_tokens = re.findall(r'\w+', source_text.lower())
    target_tokens = re.findall(r'\w+', target_text.lower())
    
    if not target_tokens:
        return 0.0
    
    source_ngrams = set(get_ngrams(source_tokens, n))
    target_ngrams = get_ngrams(target_tokens, n)
    
    novel_count = sum(1 for ngram in target_ngrams if ngram not in source_ngrams)
    
    return (novel_count / len(target_ngrams)) * 100 if target_ngrams else 0.0

def calculate_sentence_extraction(source_text, target_text):
    sentence_pattern = r'(?<=[.!?])\s+'
    source_sentences = [s.strip().lower() for s in re.split(sentence_pattern, source_text) if s.strip()]
    target_sentences = [s.strip().lower() for s in re.split(sentence_pattern, target_text) if s.strip()]
    
    if not target_sentences:
        return 0.0
    
    source_sentences_set = set(source_sentences)
    extracted_count = sum(1 for sent in target_sentences if sent in source_sentences_set)
    
    return (extracted_count / len(target_sentences)) * 100

def analyze_extractiveness(src_path, tgt_path, num_samples=200000):
    metrics = {1: [], 2: [], 3: [], 4: []}
    sent_extraction = []
    
    print(f"Analyzing extractiveness for {num_samples} samples...")
    
    try:
        with open(src_path, 'r', encoding='utf-8') as src_file, \
             open(tgt_path, 'r', encoding='utf-8') as tgt_file:
            
            for i, (src_line, tgt_line) in enumerate(zip(src_file, tgt_file)):
                if i >= num_samples:
                    break
                
                # N-gram metrics
                for n in range(1, 4): # Checking 1, 2, 3 grams
                    novel_pct = calculate_novel_ngrams(src_line, tgt_line, n)
                    metrics[n].append(novel_pct)
                
                # Sentence metric
                sent_pct = calculate_sentence_extraction(src_line, tgt_line)
                sent_extraction.append(sent_pct)
                
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1} samples...")
                    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("\n--- Extractiveness Results ---")
    print("\n[Word Level - Novel n-grams % (Lower is more extractive)]")
    for n in range(1, 4):
        avg_novel = sum(metrics[n]) / len(metrics[n]) if metrics[n] else 0
        print(f"Novel {n}-grams: {avg_novel:.2f}%")
    
    print("\n[Sentence Level - Exact Matches % (Higher is more extractive)]")
    avg_sent_ext = sum(sent_extraction) / len(sent_extraction) if sent_extraction else 0
    print(f"Extracted Sentences: {avg_sent_ext:.2f}%")
    
    # Interpretation
    avg_novel_1 = sum(metrics[1]) / len(metrics[1]) if metrics[1] else 0
    if avg_sent_ext > 70:
        print("\nInterpretation: The data is HIGHLY EXTRACTIVE (Sentence Level).")
    elif avg_sent_ext > 30:
        print("\nInterpretation: The data is MODERATELY EXTRACTIVE (Sentence Level).")
    elif avg_novel_1 < 10:
        print("\nInterpretation: The data is EXTRACTIVE (Word Level but not full sentences).")
    else:
        print("\nInterpretation: The data is highly ABSTRACTIVE.")

if __name__ == "__main__":
    src = "/Users/alberto/Desktop/Neural-Nertwork-project/nn/PNL/data/train.txt.src"
    tgt = "/Users/alberto/Desktop/Neural-Nertwork-project/nn/PNL/data/train.txt.tgt"
    analyze_extractiveness(src, tgt)
