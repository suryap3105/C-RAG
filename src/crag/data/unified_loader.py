import json
import os
from typing import List, Dict, Any

class UnifiedDatasetLoader:
    """
    Unified loader for all QA datasets.
    Maps different formats to standardized schema.
    """
    
    @staticmethod
    def load_squad(path: str) -> List[Dict[str, Any]]:
        """Load SQuAD format (v1.1 or v2.0)."""
        with open(path, 'r', encoding='utf-8') as f:
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
                        "context": context[:500],  # Truncate for memory
                        "gold_entities": []
                    })
                    idx += 1
        return standardized
    
    @staticmethod
    def load_webqsp(path: str) -> List[Dict[str, Any]]:
        """Load WebQSP Wikidata format."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        standardized = []
        for i, item in enumerate(data):
            qid = item.get('questionid', f"WebQSP.{i}")
            query = item.get('utterance', '')
            answers = item.get('answers_str', [])
            
            gold_entities = []
            if 'entities' in item:
                for e in item['entities']:
                    if 'linkings' in e:
                        labels = [link[1] for link in e['linkings'] if len(link) > 1]
                        gold_entities.extend(labels)
            
            standardized.append({
                "id": qid,
                "query": query,
                "answers": answers,
                "context": "",
                "gold_entities": list(set(gold_entities))
            })
        return standardized
    
    @staticmethod
    def load_metaqa_txt(path: str) -> List[Dict[str, Any]]:
        """Load MetaQA vanilla text format."""
        standardized = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if '\t' not in line:
                    continue
                query, answers_raw = line.split('\t', 1)
                answers = answers_raw.strip().split('|')
                
                standardized.append({
                    "id": f"MetaQA.{i}",
                    "query": query.strip(),
                    "answers": answers,
                    "context": "",
                    "gold_entities": []
                })
        return standardized
    
    @classmethod
    def load_all(cls, max_per_dataset: int = 10000) -> List[Dict[str, Any]]:
        """
        Load all available datasets with a cap per dataset.
        Returns combined standardized data.
        """
        all_data = []
        
        # SQuAD
        squad_train = "data/squad_train_v2.json"
        if os.path.exists(squad_train):
            print(f"[LOAD] SQuAD Train...")
            squad = cls.load_squad(squad_train)[:max_per_dataset]
            all_data.extend(squad)
            print(f"  -> Loaded {len(squad)} questions")
        
        # WebQSP
        webqsp_file = "data/webqsp/input/webqsp.examples.train.json"
        if os.path.exists(webqsp_file):
            print(f"[LOAD] WebQSP Train...")
            webqsp = cls.load_webqsp(webqsp_file)[:max_per_dataset]
            all_data.extend(webqsp)
            print(f"  -> Loaded {len(webqsp)} questions")
        
        print(f"\n[TOTAL] {len(all_data)} questions loaded")
        return all_data
