import re
import string
from collections import Counter
from src.base.base_metric import BaseMetric
from src.metric.rouge import AverageRougeMetric


def normalize_text(text):
    """Normalize text by removing punctuation, articles, and extra whitespace."""
    
    def remove_articles(text):
        # Remove common English articles
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def remove_punc(text):
        # Remove punctuation
        punct = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in punct)
    
    def lower(text):
        return text.lower()
    
    def whitespace_fix(text):
        tokens = text.split()
        return ' '.join([t for t in tokens if t.strip() != ''])
    
    return whitespace_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_text(prediction) == normalize_text(ground_truth)


class ExactMatch_CNNDMMetric(BaseMetric):
    """Exact match metric for CNN/DM dataset."""
    
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)
        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, model, batch):
        if 'target' not in batch:
            return 0.0
        
        targets = batch['target']
        predictions = batch['preds']
        
        exact_matches = 0.0
        total = len(predictions)
        
        for pred, target in zip(predictions, targets):
            if exact_match_score(pred, target):
                exact_matches += 1.0
        
        return 100.0 * (exact_matches / total)


class F1_CNNDMMetric(BaseMetric):
    """F1 metric for CNN/DM dataset."""
    
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)
        self.requires_preds = True
        self.compute_on_train = False
    
    def __call__(self, model, batch):
        if 'target' not in batch:
            return 0.0
        
        targets = batch['target']
        predictions = batch['preds']
        
        f1_scores = 0.0
        total = len(predictions)
        
        for pred, target in zip(predictions, targets):
            f1_scores += f1_score(pred, target)
        
        return 100.0 * (f1_scores / total)


class Rouge_CNNDMMetric(BaseMetric):
    """ROUGE metric for CNN/DM dataset."""
    
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, args, kwargs)
        self.requires_preds = True
        self.compute_on_train = False
        self.rouge_metric = AverageRougeMetric()
    
    def __call__(self, model, batch):
        if 'target' not in batch:
            return 0.0
        
        return self.rouge_metric(model, batch)
