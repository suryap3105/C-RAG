import json
import os
from typing import List, Dict, Any

class DatasetLoader:
    """
    Base loader for QA datasets.
    """
    def load(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

class SQuADLoader(DatasetLoader):
    """
    Loader for SQuAD dataset (JSON format).
    """
    def __init__(self, path: str = "data/squad_train_v2.json"):
        self.path = path

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            print(f"[WARN] Data file not found: {self.path}")
            return []
            
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        standardized = []
        idx = 0
        for article in data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                for qa in paragraph.get('qas', []):
                    qid = qa.get('id', f"SQuAD.{idx}")
                    question = qa.get('question', '')
                    
                    # Extract answers
                    answers = []
                    if not qa.get('is_impossible', False):
                        for ans in qa.get('answers', []):
                            answers.append(ans['text'])
                    
                    standardized.append({
                        "id": qid,
                        "query": question,
                        "answers": answers if answers else [""],
                        "context": context[:500],
                        "gold_entities": []
                    })
                    idx += 1
        return standardized

class WebQSPLoader(DatasetLoader):
    """
    Loader for WebQSP dataset.
    Handles official JSON format with 'utterance', 'answers_str', and 'questionid'.
    """
    def __init__(self, path: str = "data/webqsp_sample.json"):
        self.path = path

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            print(f"[WARN] Data file not found: {self.path}")
            return []
            
        with open(self.path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Support both simple list and nested 'Questions' dict if applicable
        data_list = raw_data
        if isinstance(raw_data, dict) and 'Questions' in raw_data:
            data_list = raw_data['Questions']

        standardized = []
        for i, item in enumerate(data_list):
            qid = item.get('questionid', item.get('id', f"WebQSP.{i}"))
            query = item.get('utterance', item.get('question', item.get('query', '')))
            
            # Use answers_str if available (readable), fallback to answers (IDs)
            answers = item.get('answers_str', item.get('answers', [item.get('answer', '')]))
            
            # Standardize entities
            gold_entities = []
            if 'entities' in item:
                for e in item['entities']:
                    if 'linkings' in e:
                        # Extract just the label/name from linkings [[id, label], ...]
                        labels = [link[1] for link in e['linkings'] if len(link) > 1]
                        gold_entities.extend(labels)
                    elif isinstance(e, str):
                        gold_entities.append(e)

            standardized.append({
                "id": qid,
                "query": query,
                "answers": answers,
                "gold_entities": list(set(gold_entities))
            })
        return standardized

class MetaQALoader(DatasetLoader):
    """
    Loader for MetaQA. 
    Supports JSON samples and "Vanilla" tab-separated text files.
    """
    def __init__(self, path: str = "data/metaqa_sample.json"):
        self.path = path

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            print(f"[WARN] Data file not found: {self.path}")
            return []

        # Check if it's text or JSON
        if self.path.endswith('.json'):
            with open(self.path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            standardized = []
            for i, item in enumerate(raw_data):
                qid = item.get('id', f"MetaQA.{i}")
                ans = item.get('answer', item.get('answers'))
                if isinstance(ans, str): ans = [ans]
                standardized.append({
                    "id": qid,
                    "query": item.get('question', ''),
                    "answers": ans,
                    "gold_entities": item.get('entities', [])
                })
            return standardized
        else:
            # Assume text format: question \t ans1|ans2
            standardized = []
            with open(self.path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if '\t' not in line: continue
                    query, answers_raw = line.split('\t', 1)
                    # answers_raw might have trailing \n
                    answers_raw = answers_raw.strip()
                    answers = answers_raw.split('|')
                    
                    standardized.append({
                        "id": f"MetaQA.{i}",
                        "query": query.strip(),
                        "answers": answers,
                        "gold_entities": [] # Text files typically don't have explicit labels
                    })
            return standardized
