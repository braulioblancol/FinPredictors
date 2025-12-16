#!/usr/bin/env python3
import os
import math
import random
import pickle
import re 
from tqdm import tqdm
import shutil
import gc 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ***** CHANGED: import LabelEncoder & joblib *****
from sklearn.preprocessing import LabelEncoder
import joblib

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim import AdamW

from sklearn.utils.class_weight import compute_class_weight  

import pandas as pd
import json

# -------------------------
# Seed utility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Model Definition
# -------------------------

GLOBAL_BEST_F1 = -1.0
GLOBAL_BEST_DIR = "./BEST_MODEL"
BERT_MODEL_NAME = "bert-base-multilingual-uncased"
ENCODER_FILE_NAME = "label_encoder_m1.pkl"
class MultiSegmentBertClassifier(nn.Module):
    def __init__(self,
                 bert_name=BERT_MODEL_NAME,
                 k_segments=3,
                 freeze_layers=6,
                 quantize_8bit=False,
                 num_classes_m1=10):
        super().__init__()
        self.k = k_segments

        if quantize_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.bert = AutoModel.from_pretrained(bert_name, quantization_config=bnb_config, device_map="auto")
        else:
            self.bert = AutoModel.from_pretrained(bert_name)

        hidden_size = self.bert.config.hidden_size

        # Freeze embeddings + first N layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
         

        # Segment-level processing
        self.segment_dense1 = nn.Linear(hidden_size, 512)
        self.segment_dense2 = nn.Linear(512, 256)

        concat_dim = 256 * self.k
        self.post_concat_dim1 = int(hidden_size * self.k / 2)
        log_val = math.floor(math.log2(256 * self.k))
        self.post_concat_dim2 = 2 * (log_val - 1)
        self.concat_dense1 = nn.Linear(concat_dim, self.post_concat_dim1)
        self.concat_dense2 = nn.Linear(self.post_concat_dim1, self.post_concat_dim2)

        # Two multiclass heads
        self.classifier_multi1 = nn.Linear(self.post_concat_dim2, num_classes_m1) 

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        batch_size, k, seq_len = input_ids.shape
        assert k == self.k

        flat_ids = input_ids.view(batch_size * k, seq_len)
        flat_mask = attention_mask.view(batch_size * k, seq_len)

        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask, return_dict=True)
        cls = (outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None
               else outputs.last_hidden_state[:, 0, :])

        cls = cls.view(batch_size, k, -1)

        h = self.act(self.segment_dense1(cls))
        h = self.act(self.segment_dense2(h))

        concat = h.view(batch_size, -1)
        x = self.act(self.concat_dense1(concat))
        x = self.dropout(x)
        x = self.act(self.concat_dense2(x))
        x = self.dropout(x)

        out_m1 = self.classifier_multi1(x) 
        return out_m1

# -------------------------
# Sliding window tokenizer
# -------------------------
def create_segments(text, tokenizer, max_seq_length=512, overlap=100):
    real_tokens = max_seq_length - 2
    stride = real_tokens - overlap
    tokens = tokenizer.encode(text, add_special_tokens=False)

    windows, masks = [], []
    start = 0
    while start < len(tokens):
        chunk = tokens[start:start + real_tokens]
        input_ids = tokenizer.build_inputs_with_special_tokens(chunk)
        pad_len = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attn_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_ids]
        windows.append(input_ids)
        masks.append(attn_mask)
        start += stride

    if len(windows) == 0:
        input_ids = tokenizer.build_inputs_with_special_tokens([])
        pad_len = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        windows = [input_ids]
        # mask: 1 for CLS and SEP, 0 for padding
        base_len = len(tokenizer.build_inputs_with_special_tokens([]))
        masks = [[1]*base_len + [0]*(max_seq_length - base_len)]
        
        
    return windows, masks

# -------------------------
# Dataset
# -------------------------
class MultiSegDataset(Dataset):
    def __init__(self, texts, labels_m1,
                 tokenizer_name=BERT_MODEL_NAME,
                 max_seq_length=512, overlap=100, k=3):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.k = k
        self.input_ids = []
        self.attn_masks = []
        self.labels_m1 = labels_m1 

        for text in texts:
            windows, masks = create_segments(text, self.tokenizer, max_seq_length, overlap)

            # pad / trim to exactly k
            if len(windows) < k:
                pad_win = [self.tokenizer.pad_token_id] * max_seq_length
                pad_mask = [0] * max_seq_length
                while len(windows) < k:
                    windows.append(pad_win)
                    masks.append(pad_mask)
            elif len(windows) > k:
                windows = windows[:k]
                masks = masks[:k]

            self.input_ids.append(windows)
            self.attn_masks.append(masks)

    def __len__(self): return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.attn_masks[idx], dtype=torch.long),
            torch.tensor(self.labels_m1[idx], dtype=torch.long), 
        )

def collate_fn(batch):
    ids = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    lab_m1 = torch.stack([b[2] for b in batch]) 
    return ids, masks, lab_m1

# -------------------------
# Evaluation
# -------------------------
def evaluate(model, dataloader, device):
    model.eval()
    preds_m1, trues_m1 = [], [] 

    with torch.no_grad():
        for ids, masks, lab_m1 in dataloader:
            ids, masks, lab_m1 = ids.to(device), masks.to(device), lab_m1.to(device)
            out_m1 = model(ids, masks)
            preds_m1.extend(torch.argmax(out_m1, 1).cpu().tolist())
            trues_m1.extend(lab_m1.cpu().tolist()) 

    return {
        "acc_m1": accuracy_score(trues_m1, preds_m1),
        "macro_f1_m1": f1_score(trues_m1, preds_m1, average="macro", zero_division=0), 
    }

 
 
# -------------------------
# Training
# -------------------------
def train(pkl_path,
          text_col="text_content",
          label_multi_col="page_type", 
          batch_size=8,
          epochs=5,
          lr=2e-5,
          freeze_layers=6,
          k=3,
          output_dir="./model_out",
          val_frac=0.1,
          test_frac=0.1,
          seed=42,
          quantize_8bit=False):

    metadata = {
        "bert_name": BERT_MODEL_NAME,
        "k": k,
        "freeze_layers": freeze_layers,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    texts = [row[text_col] for row in data]
    raw_m1 = [row[label_multi_col] for row in data] 

    # ***** CHANGED: Convert string → categorical *****
    le_m1 = LabelEncoder() 

    labels_m1 = le_m1.fit_transform(raw_m1) 

    # Compute weights for head 1
    class_weights_m1 = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_m1),
        y=labels_m1
    )
    class_weights_m1 = torch.tensor(class_weights_m1, dtype=torch.float) 
    
    class_weights_m1 = class_weights_m1.to(device) 

    # Save the encoders
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(le_m1, os.path.join(output_dir, ENCODER_FILE_NAME)) 

    num_classes_m1 = len(le_m1.classes_) 
    print(f"Detected classes → m1={num_classes_m1}")

    dataset = MultiSegDataset(texts, labels_m1, k=k)
    total = len(dataset)
    n_test = int(total * test_frac)
    n_val = int(total * val_frac)
    n_train = total - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiSegmentBertClassifier(
        k_segments=k, freeze_layers=freeze_layers, quantize_8bit=quantize_8bit,
        num_classes_m1=num_classes_m1
    )

    try: model.to(device)
    except: pass

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.06), num_training_steps=total_steps
    )
 
    loss_ce_m1 = nn.CrossEntropyLoss(weight=class_weights_m1) 
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = -float("inf")
    patience, no_improve = 2, 0

    # -------------------------
    # Training Loop
    # -------------------------
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}")

        for ids, masks, lab_m1 in pbar:
            ids, masks = ids.to(device), masks.to(device)
            lab_m1 = lab_m1.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out_m1 = model(ids, masks)
                loss = loss_ce_m1(out_m1, lab_m1) 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / len(train_loader):.4f}"})

        # Validation
        val = evaluate(model, val_loader, device)
        combined = val["macro_f1_m1"]
        tqdm.write(f"Epoch {ep} | loss {running_loss/len(train_loader):.4f} | combined {combined:.4f}")

        # Save best model
        if combined > best_score + 1e-4:
            best_score = combined
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            tqdm.write("Saved new best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write("Early stopping.")
                break

        # Save metadata
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
                metadata['epochs_final'] = ep
                metadata['macro_f1_m1'] = combined                  
                json.dump(metadata, f, indent=4)
                print('saving metadata')
    # -------------------------
    # Test evaluation
    # -------------------------
    best_path = os.path.join(output_dir, "best_model.pt")
    report  = None
    test = {} 
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        try: model.to(device)
        except: pass

        test = evaluate(model, test_loader, device)
        print("=== TEST RESULTS ===")
        print(test)

        # Detailed reports
        preds1, truths1 = [], []
        model.eval()
        with torch.no_grad():
            for ids, masks, lab_m1 in test_loader:
                ids, masks = ids.to(device), masks.to(device)
                out_m1 = model(ids, masks)
                preds1.extend(torch.argmax(out_m1, 1).cpu().tolist())
                truths1.extend(lab_m1.cpu().tolist())
        print("Report Head 1:")
        report = classification_report(truths1, preds1, target_names=le_m1.classes_,  digits=4)
        print(report)

    else:
        print("No best model found.")

    try:
        del model
    except:
        pass

    torch.cuda.empty_cache()
    gc.collect()    


    return {
        "lr": lr,
        "epochs": epochs,
        "freeze_layers": freeze_layers,
        "k": k,
        "batch_size": batch_size,
        "best_val_macro_f1": best_score, 
        "test_macro_f1_m1": test.get("macro_f1_m1", None),
        "output_dir": output_dir,
        "classification_report": report,
        "metadata": metadata
    }


class PageTypeInference:
    def __init__(self, model_dir=GLOBAL_BEST_DIR):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir

        # Load metadata
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        # Load tokenizer
        bert_name = self.metadata.get("bert_name", BERT_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=True)

        # Load label encoder
        self.label_encoder = joblib.load(os.path.join(model_dir, ENCODER_FILE_NAME))
        num_classes = len(self.label_encoder.classes_)
        
        self.k = self.metadata.get("k", 3)
        self.model = MultiSegmentBertClassifier(
            bert_name=bert_name,
            k_segments=self.k,
            freeze_layers=self.metadata.get("freeze_layers", 0),  # usually 0 at inference
            quantize_8bit=False,
            num_classes_m1=num_classes
        )
                # Load weights
        state = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # Optional defaults
        self.max_seq_length = 512
        self.overlap = 100


    def predict(self, text: str):
        windows, masks = create_segments(
            text,
            self.tokenizer,
            self.max_seq_length,
            self.overlap
        )

        # pad/trim
        if len(windows) < self.k:
            pad_win = [self.tokenizer.pad_token_id] * self.max_seq_length
            pad_mask = [0] * self.max_seq_length
            while len(windows) < self.k:
                windows.append(pad_win)
                masks.append(pad_mask)
        else:
            windows = windows[:self.k]
            masks = masks[:self.k]

        ids = torch.tensor([windows]).to(self.device)
        masks = torch.tensor([masks]).to(self.device)

        with torch.no_grad():
            logits = self.model(ids, masks)
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = probs.argmax().item()

        return {
            "label": self.label_encoder.inverse_transform([pred_id])[0],
            "confidence": float(probs[pred_id]),
            "probs": {
                cls: float(probs[i])
                for i, cls in enumerate(self.label_encoder.classes_)
            }
        }

    def predict_batch(self, texts: list):
        """
        Batch prediction for multiple page texts.
        Returns a list of dicts with label, confidence, and probabilities.
        """
        batch_input_ids = []
        batch_attention_masks = []

        for text in texts:
            windows, masks = create_segments(text, self.tokenizer, self.max_seq_length, self.overlap)
            
            # pad / trim to exactly self.k
            if len(windows) < self.k:
                pad_win = [self.tokenizer.pad_token_id] * self.max_seq_length
                pad_mask = [0] * self.max_seq_length
                while len(windows) < self.k:
                    windows.append(pad_win)
                    masks.append(pad_mask)
            else:
                windows = windows[:self.k]
                masks = masks[:self.k]

            batch_input_ids.append(windows)
            batch_attention_masks.append(masks)

        # convert to tensors
        ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor(batch_attention_masks, dtype=torch.long).to(self.device)

        # forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ids_tensor, masks_tensor)
            probs = torch.softmax(logits, dim=1)

        results = []
        for prob in probs:
            pred_id = prob.argmax().item()
            results.append({
                "label": self.label_encoder.inverse_transform([pred_id])[0],
                "confidence": float(prob[pred_id]),
                "probs": {cls: float(prob[i]) for i, cls in enumerate(self.label_encoder.classes_)}
            })

        return results
    
# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    pkl_path =  r"/content/drive/MyDrive/SCRIPT/labeled_data_page_type_classes2.pkl"#"/content/drive/MyDrive/SCRIPT/labeled_data_page_type.pkl"
    output_dir="/content/drive/MyDrive/SCRIPT/pt_lang_model_out_definitive"
    output_dir2="/content/drive/MyDrive/SCRIPT/stats_definitive"
    runs_dir="/content/drive/MyDrive/SCRIPT/runs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    parameters_lr = [5e-5,2e-5,1e-5]
    parameters_epoch = [5,10,15]
    freeze_layers = [4,6,8]
    k=[2,3,4]
    batch_size = [8,10,12]

    all_results = [] 
    
    for lr in parameters_lr:
        for epoch in parameters_epoch:
            for fl in freeze_layers:
                for ks in k:
                    for bs in batch_size:
                        print(f"Training with lr={lr}, epochs={epoch}, freeze_layers={fl}, k={ks}, batch_size={bs}")
                        RUN_DIR = f"{runs_dir}/lr{lr}_ep{epoch}_fl{fl}_k{ks}_bs{bs}"                        
                        os.makedirs(runs_dir, exist_ok=True)
                        res = train(
                            pkl_path,
                            text_col="text_content",
                            label_multi_col="page_type", 
                            batch_size=bs,
                            epochs=epoch,
                            lr=lr,
                            freeze_layers=fl,
                            k=ks,
                            quantize_8bit=False,
                            output_dir=RUN_DIR
                        )
                        all_results.append(res)

                        if res["best_val_macro_f1"] > GLOBAL_BEST_F1:

                            GLOBAL_BEST_F1 = res["best_val_macro_f1"] 

                            # replace
                            shutil.copy(
                                os.path.join(res["output_dir"], "best_model.pt"),
                                os.path.join(output_dir, "best_model.pt")
                            )

                            shutil.copy(
                                os.path.join(res["output_dir"], ENCODER_FILE_NAME),
                                os.path.join(output_dir, ENCODER_FILE_NAME)
                            )

                            shutil.copy(
                                os.path.join(res["output_dir"], "meta.json"),
                                os.path.join(output_dir, "meta.json")
                            )

                            print("🔥 New GLOBAL BEST model found")

    
                        if os.path.exists(RUN_DIR) and os.path.isdir(RUN_DIR):
                            shutil.rmtree(RUN_DIR)
    print("Training complete.")

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir2 + "/experiment_results.csv", index=False)

    with open(output_dir2+ "/experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("Saved results to experiment_results.csv and experiment_results.json")