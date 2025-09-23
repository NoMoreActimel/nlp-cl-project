
from src.base.base_metric import BaseMetric
from src.metric.mt5_mlqa_metrics import mt5_mlqa_em, mt5_mlqa_f1
from src.metric.orig_mlqa_metrics import metric_max_over_ground_truths, exact_match_score, f1_score


class ExactMatch_MLQAMetric(BaseMetric):
    def __init__(self, name=None, model_type="enc-dec", use_mt5_code=True, *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.model_type = model_type
        self.requires_preds = True
        self.compute_on_train = False

        self.use_mt5_code = use_mt5_code

    def _orig_mlqa_em(self, predictions, answers, langs):
        exact_match = 0.0
        total = len(predictions)
        for pred, answer, lang in zip(predictions, answers, langs):
            if not isinstance(answer, list):
                answer = [answer]
            exact_match += metric_max_over_ground_truths(exact_match_score, pred, answer, lang) 
        exact_match = exact_match / total
        return 100 * exact_match

    def __call__(self, model, batch):
        if ('answer' not in batch) or ('lang' not in batch):
            return 0.0
        
        answers, predictions, langs = batch['answer'], batch['preds'], batch['lang']

        if self.model_type == "dec":
            predictions = [pred.split("answer:")[-1].strip() for pred in predictions]
        
        if self.use_mt5_code:
            return mt5_mlqa_em(answers, predictions, langs[0])
        return self._orig_mlqa_em(predictions, answers, langs)

class F1_MLQAMetric(BaseMetric):
    def __init__(self, name=None, model_type="enc-dec", use_mt5_code=True, *args, **kwargs):
        super().__init__(name, args, kwargs)

        self.model_type = model_type
        self.requires_preds = True
        self.compute_on_train = False

        self.use_mt5_code = use_mt5_code

    def _orig_mlqa_f1(self, predictions, answers, langs):
        f1 = 0.0
        total = len(predictions)
        for pred, answer, lang in zip(predictions, answers, langs):
            if not isinstance(answer, list):
                answer = [answer]
            f1 += metric_max_over_ground_truths(f1_score, pred, answer, lang)
        f1 = f1 / total
        return 100 * f1
    
    def __call__(self, model, batch):
        if ('answer' not in batch) or ('lang' not in batch):
            return 0.0

        answers, predictions, langs = batch['answer'], batch['preds'], batch['lang']

        if self.model_type == "dec":
            # answers = [ans.split("answer:")[-1].strip() for ans in answers]
            predictions = [
                [pred.split("answer:")[-1].strip() for pred in sample_preds]
                for sample_preds in predictions
            ]
        
        if self.use_mt5_code:
            return mt5_mlqa_f1(answers, predictions, langs[0])
        return self._orig_mlqa_f1(predictions, answers, langs)
