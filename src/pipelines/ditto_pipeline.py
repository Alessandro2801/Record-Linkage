#!/usr/bin/env python3
"""
ditto_pipeline.py
Pipeline per Entity Matching con Ditto - Auto-ottimizzata per qualsiasi GPU

Rileva automaticamente le caratteristiche della GPU e configura i parametri
ottimali per sfruttare al massimo la memoria e le capacità disponibili.

Usage:
    python ditto_pipeline.py --base-path /path/to/project --mode train
    python ditto_pipeline.py --base-path /path/to/project --mode evaluate
    python ditto_pipeline.py --base-path /path/to/project --mode inference
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AUTO-CONFIGURAZIONE GPU
# ============================================================================

def get_gpu_info() -> dict:
    """Rileva informazioni sulla GPU disponibile."""
    info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'name': 'CPU',
        'memory_gb': 0,
        'compute_capability': (0, 0),
        'supports_bf16': False,
        'supports_tf32': False
    }
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            'device_count': torch.cuda.device_count(),
            'name': props.name,
            'memory_gb': props.total_memory / 1e9,
            'compute_capability': (props.major, props.minor),
            'supports_bf16': props.major >= 8,  # Ampere+
            'supports_tf32': props.major >= 8,  # Ampere+
        })
    
    return info


@dataclass
class DittoConfig:
    """
    Configurazione auto-ottimizzata per Entity Matching con Ditto.
    Si adatta automaticamente alla GPU disponibile per massimizzare le performance.
    """
    # GPU Settings (auto-detected)
    device: str = "cuda"
    fp16: bool = True
    bf16: bool = False
    
    # Model (auto-selected based on VRAM)
    model_name: str = "roberta-base"
    
    # Batch Size (auto-scaled based on VRAM)
    train_batch_size: int = 64
    eval_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    
    # Sequence Length
    max_length: int = 128
    
    # Training Hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 10
    
    # Data Augmentation (Ditto optimizations)
    use_data_augmentation: bool = True
    da_operators: List[str] = field(default_factory=lambda: ['del', 'swap', 'drop_col'])
    augmentation_prob: float = 0.4
    
    # Domain Knowledge Injection
    use_domain_knowledge: bool = True
    
    # Text Summarization
    use_summarization: bool = True
    summarize_threshold: int = 200
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Early Stopping
    early_stopping_patience: int = 3
    
    # DataLoader
    num_workers: int = 4
    
    # Output
    output_dir: str = "models/ditto"
    
    def __post_init__(self):
        """Auto-configura in base alla GPU rilevata."""
        gpu_info = get_gpu_info()
        
        print("=" * 60)
        print("🔍 AUTO-DETECTION HARDWARE")
        print("=" * 60)
        
        if not gpu_info['available']:
            print("⚠️  Nessuna GPU rilevata - uso CPU")
            self.device = "cpu"
            self.fp16 = False
            self.bf16 = False
            self.train_batch_size = 8
            self.eval_batch_size = 16
            self.model_name = "distilbert-base-uncased"
            self.max_length = 64
            self.num_workers = 2
            return
        
        vram = gpu_info['memory_gb']
        name = gpu_info['name']
        cc = gpu_info['compute_capability']
        
        print(f"🖥️  GPU: {name}")
        print(f"💾 VRAM: {vram:.1f} GB")
        print(f"🔧 Compute Capability: {cc[0]}.{cc[1]}")
        
        # ====================================================================
        # CONFIGURAZIONE AUTOMATICA PER TIER DI GPU
        # ====================================================================
        
        # TIER 1: Entry Level (< 8GB) - GTX 1060, RTX 3050, etc.
        if vram < 8:
            self._configure_tier1()
        
        # TIER 2: Consumer (8-16GB) - RTX 3070, 3080, 4070
        elif vram < 16:
            self._configure_tier2()
        
        # TIER 3: Prosumer (16-24GB) - RTX 3090, 4090, A5000
        elif vram < 24:
            self._configure_tier3()
        
        # TIER 4: Professional (24-48GB) - A6000, A40, L40
        elif vram < 48:
            self._configure_tier4()
        
        # TIER 5: Datacenter (48-80GB) - A100, H100 80GB
        elif vram < 80:
            self._configure_tier5()
        
        # TIER 6: HBM Datacenter (80GB+) - H100 80GB, H200, MI300X
        else:
            self._configure_tier6()
        
        # BF16 se supportato (Ampere+)
        if gpu_info['supports_bf16']:
            self.bf16 = True
            print(f"✅ BFloat16 abilitato (Compute >= 8.0)")
        
        # TF32 se supportato
        if gpu_info['supports_tf32']:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"✅ TF32 abilitato per CUDA matmul")
        
        # cuDNN benchmark
        torch.backends.cudnn.benchmark = True
        
        print("-" * 60)
        print(f"📊 CONFIGURAZIONE SELEZIONATA:")
        print(f"   Modello: {self.model_name}")
        print(f"   Batch Size: {self.train_batch_size} (train) / {self.eval_batch_size} (eval)")
        print(f"   Max Length: {self.max_length}")
        print(f"   Workers: {self.num_workers}")
        print(f"   FP16: {self.fp16} | BF16: {self.bf16}")
        print("=" * 60)
    
    def _configure_tier1(self):
        """< 8GB VRAM - Entry level"""
        print("📱 Tier 1: Entry Level GPU")
        self.model_name = "distilbert-base-uncased"
        self.train_batch_size = 16
        self.eval_batch_size = 32
        self.max_length = 64
        self.num_workers = 2
        self.gradient_accumulation_steps = 2
    
    def _configure_tier2(self):
        """8-16GB VRAM - Consumer"""
        print("🎮 Tier 2: Consumer GPU")
        self.model_name = "distilbert-base-uncased"
        self.train_batch_size = 32
        self.eval_batch_size = 64
        self.max_length = 128
        self.num_workers = 4
    
    def _configure_tier3(self):
        """16-24GB VRAM - Prosumer"""
        print("💪 Tier 3: Prosumer GPU")
        self.model_name = "roberta-base"
        self.train_batch_size = 64
        self.eval_batch_size = 128
        self.max_length = 128
        self.num_workers = 4
    
    def _configure_tier4(self):
        """24-48GB VRAM - Professional"""
        print("🏢 Tier 4: Professional GPU")
        self.model_name = "roberta-large"
        self.train_batch_size = 96
        self.eval_batch_size = 192
        self.max_length = 192
        self.num_workers = 6
    
    def _configure_tier5(self):
        """48-80GB VRAM - Datacenter"""
        print("🏭 Tier 5: Datacenter GPU")
        self.model_name = "roberta-large"
        self.train_batch_size = 192
        self.eval_batch_size = 384
        self.max_length = 256
        self.num_workers = 8
    
    def _configure_tier6(self):
        """80GB+ VRAM - HBM Datacenter (H100/H200/MI300X)"""
        print("🚀 Tier 6: HBM Datacenter GPU - MASSIME PERFORMANCE")
        self.model_name = "roberta-large"  # Oppure deberta-v3-large
        self.train_batch_size = 256
        self.eval_batch_size = 512
        self.max_length = 256
        self.num_workers = 8
        # Opzionale: modelli ancora più grandi
        # self.model_name = "microsoft/deberta-v3-large"


# ============================================================================
# DATA AUGMENTATION (Stile Ditto)
# ============================================================================

class DittoDataAugmentor:
    """
    Implementa le tecniche di Data Augmentation di Ditto:
    - del: elimina token casuali
    - swap: scambia token adiacenti  
    - drop_col: elimina un'intera colonna
    """
    
    def __init__(self, operators: List[str], prob: float = 0.4):
        self.operators = operators
        self.prob = prob
        
    def augment(self, text: str) -> str:
        """Applica augmentation casuale al testo."""
        if random.random() > self.prob:
            return text
            
        operator = random.choice(self.operators)
        
        if operator == 'del':
            return self._delete_tokens(text)
        elif operator == 'swap':
            return self._swap_tokens(text)
        elif operator == 'drop_col':
            return self._drop_column(text)
        else:
            return text
    
    def _delete_tokens(self, text: str, del_ratio: float = 0.1) -> str:
        """Elimina token casuali."""
        tokens = text.split()
        n_del = max(1, int(len(tokens) * del_ratio))
        if len(tokens) <= n_del:
            return text
        indices = random.sample(range(len(tokens)), n_del)
        return ' '.join([t for i, t in enumerate(tokens) if i not in indices])
    
    def _swap_tokens(self, text: str) -> str:
        """Scambia coppie di token adiacenti."""
        tokens = text.split()
        if len(tokens) < 2:
            return text
        idx = random.randint(0, len(tokens) - 2)
        tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
        return ' '.join(tokens)
    
    def _drop_column(self, text: str) -> str:
        """Elimina un'intera colonna (formato COL ... VAL ...)."""
        # Trova pattern COL xxx VAL yyy
        parts = text.split(' COL ')
        if len(parts) <= 2:
            return text
        # Rimuovi una colonna casuale (non la prima)
        idx = random.randint(1, len(parts) - 1)
        parts.pop(idx)
        return ' COL '.join(parts)


# ============================================================================
# DOMAIN KNOWLEDGE INJECTION
# ============================================================================

class VehicleDomainKnowledge:
    """
    Inietta conoscenza di dominio specifica per i veicoli.
    Evidenzia informazioni chiave per il matching.
    """
    
    # Abbreviazioni comuni nel dominio auto
    ABBREVIATIONS = {
        'chev': 'chevrolet',
        'chevy': 'chevrolet',
        'vw': 'volkswagen',
        'merc': 'mercedes',
        'bmw': 'bayerische motoren werke',
        'gmc': 'general motors company',
        'suv': 'sport utility vehicle',
        'mpv': 'multi purpose vehicle',
        '4wd': 'four wheel drive',
        'awd': 'all wheel drive',
        'fwd': 'front wheel drive',
        'rwd': 'rear wheel drive',
        'auto': 'automatic',
        'man': 'manual',
        'cyl': 'cylinders',
    }
    
    # Sinonimi per body type
    BODY_SYNONYMS = {
        'sedan': ['sedan', 'saloon', '4-door', '4dr'],
        'suv': ['suv', 'sport utility', 'crossover', 'cuv'],
        'truck': ['truck', 'pickup', 'pick-up', 'ute'],
        'coupe': ['coupe', 'coupé', '2-door', '2dr'],
        'hatchback': ['hatchback', 'hatch', '5-door', '5dr'],
        'wagon': ['wagon', 'estate', 'touring', 'avant'],
        'van': ['van', 'minivan', 'mpv', 'people carrier'],
        'convertible': ['convertible', 'cabriolet', 'cabrio', 'roadster'],
    }
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalizza abbreviazioni e sinonimi."""
        text_lower = text.lower()
        for abbr, full in cls.ABBREVIATIONS.items():
            text_lower = text_lower.replace(f' {abbr} ', f' {full} ')
            text_lower = text_lower.replace(f' {abbr},', f' {full},')
        return text_lower
    
    @classmethod
    def add_markers(cls, record: dict) -> dict:
        """
        Aggiunge marker [IMPORTANT] ai campi discriminanti.
        Ditto usa questi marker per focalizzare l'attenzione del modello.
        """
        important_fields = ['manufacturer', 'model', 'year', 'price', 'mileage']
        
        marked_record = record.copy()
        for field in important_fields:
            if field in marked_record and marked_record[field]:
                marked_record[field] = f"[IMP] {marked_record[field]} [/IMP]"
        
        return marked_record


# ============================================================================
# TEXT SUMMARIZATION
# ============================================================================

class TextSummarizer:
    """
    Riassume testi troppo lunghi mantenendo solo token ad alto TF-IDF.
    Implementazione semplificata del summarizer di Ditto.
    """
    
    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'col', 'val', 'none', 'null', 'nan', ''
        }
    
    def summarize(self, text: str) -> str:
        """Riassume il testo se supera max_length."""
        if len(text) <= self.max_length:
            return text
        
        # Tokenizza e filtra stopwords
        tokens = text.lower().split()
        
        # Calcola frequenza (proxy per TF-IDF semplificato)
        freq = {}
        for token in tokens:
            if token not in self.stopwords:
                freq[token] = freq.get(token, 0) + 1
        
        # Ordina per frequenza inversa (token rari = più informativi)
        sorted_tokens = sorted(freq.keys(), key=lambda x: freq[x])
        
        # Ricostruisci mantenendo ordine originale e token importanti
        important = set(sorted_tokens[:len(sorted_tokens)//2])
        result_tokens = []
        
        for token in tokens:
            if token in important or token in ['col', 'val', '[imp]', '[/imp]']:
                result_tokens.append(token)
            if len(' '.join(result_tokens)) >= self.max_length:
                break
        
        return ' '.join(result_tokens)


# ============================================================================
# DATASET
# ============================================================================

class DittoDataset(Dataset):
    """
    Dataset per Entity Matching in stile Ditto.
    Serializza coppie di record in formato:
    [CLS] COL col1 VAL val1 COL col2 VAL val2 ... [SEP] COL col1 VAL val1 ... [SEP]
    """
    
    def __init__(
        self, 
        pairs: List[Tuple[dict, dict, int]],
        tokenizer,
        max_length: int = 256,
        augmentor: Optional[DittoDataAugmentor] = None,
        summarizer: Optional[TextSummarizer] = None,
        use_domain_knowledge: bool = True
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentor = augmentor
        self.summarizer = summarizer
        self.use_domain_knowledge = use_domain_knowledge
        
    def __len__(self):
        return len(self.pairs)
    
    def serialize_record(self, record: dict) -> str:
        """Serializza un record in formato Ditto."""
        if self.use_domain_knowledge:
            record = VehicleDomainKnowledge.add_markers(record)
            
        parts = []
        for col, val in record.items():
            if val is not None and str(val).strip() and str(val).lower() not in ['nan', 'none', '']:
                val_str = str(val).strip()
                if self.use_domain_knowledge:
                    val_str = VehicleDomainKnowledge.normalize(val_str)
                parts.append(f"COL {col} VAL {val_str}")
        
        text = ' '.join(parts)
        
        # Augmentation (solo durante training)
        if self.augmentor:
            text = self.augmentor.augment(text)
        
        # Summarization se troppo lungo
        if self.summarizer:
            text = self.summarizer.summarize(text)
            
        return text
    
    def __getitem__(self, idx):
        record_a, record_b, label = self.pairs[idx]
        
        text_a = self.serialize_record(record_a)
        text_b = self.serialize_record(record_b)
        
        # Tokenizza come sequence pair
        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carica e prepara i dati."""
    print("📂 Caricamento dati...")
    
    mediated_path = os.path.join(base_path, 'data/mediated_schema/mediated_schema_normalized.csv')
    train_path = os.path.join(base_path, 'data/ground_truth/GT_train/train.csv')
    val_path = os.path.join(base_path, 'data/ground_truth/GT_train/val.csv')
    test_path = os.path.join(base_path, 'data/ground_truth/GT_train/test.csv')
    
    df = pd.read_csv(mediated_path, dtype={'id_source_vehicles': 'object'}, low_memory=False)
    gt_train = pd.read_csv(train_path)
    gt_val = pd.read_csv(val_path)
    gt_test = pd.read_csv(test_path)
    
    # Crea ID unificato e imposta come indice
    df['id_unificato'] = df['id_source_vehicles'].fillna(df['id_source_used_cars'])
    df = df.set_index('id_unificato')
    
    # Rimuovi colonne non necessarie
    cols_to_drop = ['vin', 'description', 'id_source_vehicles', 'id_source_used_cars']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    print(f"   Dataset: {len(df):,} record")
    print(f"   Train: {len(gt_train):,} coppie")
    print(f"   Val: {len(gt_val):,} coppie")
    print(f"   Test: {len(gt_test):,} coppie")
    
    return df, gt_train, gt_val, gt_test


def create_pairs_from_gt(df: pd.DataFrame, gt: pd.DataFrame) -> List[Tuple[dict, dict, int]]:
    """Crea lista di (record_a, record_b, label) dalla ground truth."""
    pairs = []
    skipped = 0
    
    for _, row in gt.iterrows():
        id_a = str(row['id_A'])
        id_b = str(row['id_B'])
        
        if id_a in df.index and id_b in df.index:
            record_a = df.loc[id_a].to_dict()
            record_b = df.loc[id_b].to_dict()
            label = int(row['label'])
            pairs.append((record_a, record_b, label))
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"   ⚠️  Saltate {skipped} coppie (ID non trovati)")
    
    return pairs


# ============================================================================
# TRAINING
# ============================================================================

class DittoTrainer:
    """Trainer auto-ottimizzato per Entity Matching."""
    
    def __init__(self, config: DittoConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model e tokenizer
        print(f"🤖 Caricamento modello: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2
        ).to(self.device)
        
        # Abilita ottimizzazioni CUDA
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.fp16 else None
        
        # Augmentor e Summarizer
        self.augmentor = DittoDataAugmentor(
            config.da_operators, 
            config.augmentation_prob
        ) if config.use_data_augmentation else None
        
        self.summarizer = TextSummarizer(
            config.summarize_threshold
        ) if config.use_summarization else None
        
        # Metriche
        self.best_f1 = 0.0
        self.patience_counter = 0
        
    def train(self, train_pairs: List, val_pairs: List):
        """Training loop principale."""
        print(f"\n{'='*60}")
        print("🚀 AVVIO TRAINING DITTO")
        print(f"{'='*60}")
        print(f"📊 Configurazione:")
        print(f"   - Modello: {self.config.model_name}")
        print(f"   - Batch size: {self.config.train_batch_size}")
        print(f"   - Max length: {self.config.max_length}")
        print(f"   - Epochs: {self.config.num_epochs}")
        print(f"   - Learning rate: {self.config.learning_rate}")
        print(f"   - Data Augmentation: {self.config.use_data_augmentation}")
        print(f"   - Domain Knowledge: {self.config.use_domain_knowledge}")
        print(f"   - FP16: {self.config.fp16}")
        
        # Dataset
        train_dataset = DittoDataset(
            train_pairs, 
            self.tokenizer,
            self.config.max_length,
            augmentor=self.augmentor,
            summarizer=self.summarizer,
            use_domain_knowledge=self.config.use_domain_knowledge
        )
        
        val_dataset = DittoDataset(
            val_pairs,
            self.tokenizer, 
            self.config.max_length,
            augmentor=None,  # No augmentation in validation
            summarizer=self.summarizer,
            use_domain_knowledge=self.config.use_domain_knowledge
        )
        
        # DataLoader con pin_memory per GPU
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        # Optimizer con weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_params,
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\n📈 Training steps: {total_steps:,} (warmup: {warmup_steps:,})")
        print(f"📦 Batches per epoch: {len(train_loader):,}")
        
        # Training loop
        global_step = 0
        train_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'─'*60}")
            print(f"📅 Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'─'*60}")
            
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc="Training", leave=True)
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass con mixed precision
                if self.config.fp16:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # Più efficiente
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (progress_bar.n + 1)
                    self.logger.info(f"Step {global_step}: loss={avg_loss:.4f}")
            
            # Evaluation alla fine di ogni epoch
            print("\n🔍 Evaluating...")
            metrics = self.evaluate(val_loader)
            
            print(f"   📊 Validation Results:")
            print(f"      F1: {metrics['f1']:.4f}")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall: {metrics['recall']:.4f}")
            print(f"      Accuracy: {metrics['accuracy']:.4f}")
            
            # Early stopping check
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                self.patience_counter = 0
                self._save_checkpoint(epoch, metrics)
                print(f"   ✅ Nuovo best F1! Modello salvato.")
            else:
                self.patience_counter += 1
                print(f"   ⏳ Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                    break
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETATO!")
        print(f"   Best F1: {self.best_f1:.4f}")
        print(f"{'='*60}")
        
        return self.best_f1
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Valutazione del modello."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels']
            
            if self.config.fp16:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        
        return {
            'f1': f1_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'accuracy': accuracy_score(all_labels, all_preds)
        }
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Salva checkpoint del modello."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Salva modello
        model_path = os.path.join(self.config.output_dir, 'best_model')
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Salva config e metriche
        checkpoint_info = {
            'epoch': epoch,
            'metrics': metrics,
            'config': {
                'model_name': self.config.model_name,
                'max_length': self.config.max_length,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.train_batch_size
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.config.output_dir, 'checkpoint_info.json'), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Carica modello da checkpoint."""
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"✅ Modello caricato da: {checkpoint_path}")


# ============================================================================
# INFERENCE
# ============================================================================

class DittoMatcher:
    """Matcher per inference su nuove coppie."""
    
    def __init__(self, model_path: str, config: DittoConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        print(f"📂 Caricamento modello da: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.summarizer = TextSummarizer(config.summarize_threshold)
    
    def serialize_record(self, record: dict) -> str:
        """Serializza un record."""
        record = VehicleDomainKnowledge.add_markers(record)
        parts = []
        for col, val in record.items():
            if val is not None and str(val).strip() and str(val).lower() not in ['nan', 'none', '']:
                val_str = VehicleDomainKnowledge.normalize(str(val).strip())
                parts.append(f"COL {col} VAL {val_str}")
        text = ' '.join(parts)
        return self.summarizer.summarize(text)
    
    @torch.no_grad()
    def predict(self, pairs: List[Tuple[dict, dict]]) -> List[Tuple[float, int]]:
        """
        Predice se le coppie sono match.
        Ritorna lista di (probability, prediction).
        """
        results = []
        
        # Batch processing
        batch_size = self.config.eval_batch_size
        
        for i in tqdm(range(0, len(pairs), batch_size), desc="Predicting"):
            batch_pairs = pairs[i:i + batch_size]
            
            # Serializza e tokenizza
            texts_a = [self.serialize_record(p[0]) for p in batch_pairs]
            texts_b = [self.serialize_record(p[1]) for p in batch_pairs]
            
            encoding = self.tokenizer(
                texts_a,
                texts_b,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with autocast() if self.config.fp16 else torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            for prob, pred in zip(probs, preds):
                results.append((float(prob), int(pred)))
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ditto Entity Matching - Auto-Optimized')
    parser.add_argument('--base-path', type=str, required=True,
                        help='Path base del progetto')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'],
                        default='train', help='Modalità di esecuzione')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path al modello salvato (per evaluate/inference)')
    args = parser.parse_args()
    
    # Configurazione auto-ottimizzata
    config = DittoConfig()
    config.output_dir = os.path.join(args.base_path, 'models/ditto')
    
    if args.mode == 'train':
        # Carica dati
        df, gt_train, gt_val, gt_test = load_and_prepare_data(args.base_path)
        
        # Crea coppie
        print("\n📋 Creazione coppie di training...")
        train_pairs = create_pairs_from_gt(df, gt_train)
        val_pairs = create_pairs_from_gt(df, gt_val)
        
        print(f"   Train pairs: {len(train_pairs):,}")
        print(f"   Val pairs: {len(val_pairs):,}")
        
        # Bilanciamento classi
        pos = sum(1 for _, _, l in train_pairs if l == 1)
        neg = len(train_pairs) - pos
        print(f"   Match/Non-match ratio: {pos}/{neg} ({pos/len(train_pairs)*100:.1f}%)")
        
        # Training
        trainer = DittoTrainer(config)
        best_f1 = trainer.train(train_pairs, val_pairs)
        
        # Test finale
        print("\n📊 Valutazione finale su Test Set...")
        test_pairs = create_pairs_from_gt(df, gt_test)
        test_dataset = DittoDataset(
            test_pairs,
            trainer.tokenizer,
            config.max_length,
            use_domain_knowledge=config.use_domain_knowledge
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Carica best model
        best_model_path = os.path.join(config.output_dir, 'best_model')
        trainer.load_checkpoint(best_model_path)
        
        test_metrics = trainer.evaluate(test_loader)
        print(f"\n{'='*60}")
        print(f"🏆 RISULTATI FINALI (Test Set)")
        print(f"{'='*60}")
        print(f"   F1 Score:  {test_metrics['f1']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        
    elif args.mode == 'evaluate':
        model_path = args.model_path or os.path.join(config.output_dir, 'best_model')
        
        df, _, _, gt_test = load_and_prepare_data(args.base_path)
        test_pairs = create_pairs_from_gt(df, gt_test)
        
        matcher = DittoMatcher(model_path, config)
        
        # Prepara dataset
        test_dataset = DittoDataset(
            test_pairs,
            matcher.tokenizer,
            config.max_length,
            use_domain_knowledge=config.use_domain_knowledge
        )
        test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size)
        
        trainer = DittoTrainer(config)
        trainer.model = matcher.model
        trainer.tokenizer = matcher.tokenizer
        
        metrics = trainer.evaluate(test_loader)
        print(f"\n📊 Test Results:")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
    elif args.mode == 'inference':
        model_path = args.model_path or os.path.join(config.output_dir, 'best_model')
        
        # Esempio inference
        print("⚠️  Modalità inference - implementa la tua logica qui")
        print(f"   Modello: {model_path}")


if __name__ == '__main__':
    main()
