"""
Enhanced NER Model: BERT + BiLSTM + CRF
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class BertBiLSTMCRF(nn.Module):
    """
    BERT encoder + BiLSTM + CRF for token classification
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        use_crf: bool = True,
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Classifier
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_labels)
        
        # CRF layer
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        sequence_output = self.dropout(sequence_output)
        
        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # (batch, seq_len, lstm_hidden*2)
        lstm_output = self.dropout(lstm_output)
        
        # Classifier
        logits = self.classifier(lstm_output)  # (batch, seq_len, num_labels)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            if self.use_crf:
                # CRF loss
                # Mask padded tokens
                mask = attention_mask.bool()
                # CRF expects log_likelihood
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            else:
                # Standard cross-entropy
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            
            outputs["loss"] = loss
        
        return outputs
    
    def predict(self, input_ids, attention_mask=None):
        """Predict with CRF decoding"""
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]
        
        if self.use_crf:
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            # Pad predictions to max length
            max_len = logits.size(1)
            padded_preds = []
            for pred in predictions:
                padded_preds.append(pred + [0] * (max_len - len(pred)))
            return torch.tensor(padded_preds, device=logits.device)
        else:
            return logits.argmax(dim=-1)


def create_model(model_name: str, num_labels: int, use_lstm: bool = True, use_crf: bool = True):
    """
    Create model based on configuration
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of labels
        use_lstm: Whether to use BiLSTM layer
        use_crf: Whether to use CRF layer
    """
    if use_lstm:
        return BertBiLSTMCRF(
            model_name=model_name,
            num_labels=num_labels,
            lstm_hidden_dim=128,
            lstm_layers=1,
            dropout=0.3,
            use_crf=use_crf,
        )
    else:
        # Fallback to standard BERT token classifier
        from transformers import AutoModelForTokenClassification
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )