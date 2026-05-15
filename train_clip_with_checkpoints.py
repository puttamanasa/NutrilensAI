# """
# train_clip_with_checkpoints.py
# ─────────────────────────────
# Fine-tunes CLIP on your food20 Indian dataset.
# Saves a checkpoint after every epoch so you can
# resume training at any point.

# Run:
#     python train_clip_with_checkpoints.py
#     python train_clip_with_checkpoints.py --resume checkpoints/epoch_3
# """

# import os
# import random
# import argparse
# import torch
# from datasets import load_dataset
# from transformers import CLIPProcessor, CLIPModel
# from torch.utils.data import DataLoader

# # ─────────────────────────────────────────────
# # CONFIG  (edit these paths as needed)
# # ─────────────────────────────────────────────
# DATA_DIR        = "food20dataset"        # folder with class sub-folders
# CHECKPOINT_DIR  = "checkpoints"          # where checkpoints are saved
# FINAL_MODEL_DIR = "clip-food20-indian"   # final saved model
# EPOCHS          = 5
# BATCH_SIZE      = 8
# LR              = 1e-5
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# # ─────────────────────────────────────────────
# # ARGUMENT PARSER
# # ─────────────────────────────────────────────
# parser = argparse.ArgumentParser()
# parser.add_argument("--resume", type=str, default=None,
#                     help="Path to checkpoint folder to resume from (e.g. checkpoints/epoch_3)")
# args = parser.parse_args()


# # ─────────────────────────────────────────────
# # DEVICE
# # ─────────────────────────────────────────────
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")


# # ─────────────────────────────────────────────
# # DATASET
# # ─────────────────────────────────────────────
# print("Loading dataset …")
# dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
# labels  = dataset["train"].features["label"].names
# print(f"Classes ({len(labels)}): {labels}")


# def preprocess(example):
#     label_name = labels[example["label"]]
#     prompts = [
#         f"a photo of {label_name}",
#         f"a delicious indian dish {label_name}",
#         f"a plate of {label_name}",
#         f"authentic indian food {label_name}"
#     ]
#     return {"image": example["image"], "text": random.choice(prompts)}

# dataset = dataset.map(preprocess)


# # ─────────────────────────────────────────────
# # MODEL & PROCESSOR
# # ─────────────────────────────────────────────
# if args.resume:
#     print(f"Resuming from checkpoint: {args.resume}")
#     model     = CLIPModel.from_pretrained(args.resume).to(device)
#     processor = CLIPProcessor.from_pretrained(args.resume)
# else:
#     print("Loading base CLIP model …")
#     model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Freeze backbone — only train projection heads
# for param in model.vision_model.parameters():
#     param.requires_grad = False
# for param in model.text_model.parameters():
#     param.requires_grad = False


# # ─────────────────────────────────────────────
# # COLLATE  (run processor per batch)
# # ─────────────────────────────────────────────
# def collate_fn(examples):
#     images = [ex["image"] for ex in examples]
#     texts  = [ex["text"]  for ex in examples]
#     labels_batch = [ex["label"] for ex in examples]

#     inputs = processor(
#         text=texts, images=images,
#         return_tensors="pt", padding=True, truncation=True
#     )
#     inputs["labels"] = torch.tensor(labels_batch)
#     return inputs


# train_loader = DataLoader(
#     dataset["train"],
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     collate_fn=collate_fn
# )


# # ─────────────────────────────────────────────
# # OPTIMIZER  (resume epoch counter if needed)
# # ─────────────────────────────────────────────
# optimizer   = torch.optim.AdamW(model.parameters(), lr=LR)
# start_epoch = 0

# # Read the last saved epoch number from checkpoint folder name
# if args.resume:
#     try:
#         start_epoch = int(os.path.basename(args.resume).split("_")[-1])
#         print(f"Resuming from epoch {start_epoch}")
#     except ValueError:
#         pass


# # ─────────────────────────────────────────────
# # TRAINING LOOP
# # ─────────────────────────────────────────────
# model.train()

# for epoch in range(start_epoch, EPOCHS):
#     total_loss = 0.0

#     for step, batch in enumerate(train_loader):
#         optimizer.zero_grad()

#         outputs = model(
#             input_ids      = batch["input_ids"].to(device),
#             attention_mask = batch["attention_mask"].to(device),
#             pixel_values   = batch["pixel_values"].to(device),
#             return_loss    = True
#         )

#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         # Print progress every 20 steps
#         if (step + 1) % 20 == 0:
#             print(f"  Epoch {epoch+1}/{EPOCHS}  Step {step+1}/{len(train_loader)}"
#                   f"  Loss: {loss.item():.4f}")

#     avg_loss = total_loss / len(train_loader)
#     print(f"\n✅ Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}\n")

#     # ── SAVE CHECKPOINT ──────────────────────────────────────────────────
#     ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
#     model.save_pretrained(ckpt_path)
#     processor.save_pretrained(ckpt_path)
#     print(f"💾 Checkpoint saved → {ckpt_path}\n")


# # ─────────────────────────────────────────────
# # SAVE FINAL MODEL
# # ─────────────────────────────────────────────
# model.save_pretrained(FINAL_MODEL_DIR)
# processor.save_pretrained(FINAL_MODEL_DIR)
# print(f"\n🎉 Training complete! Final model saved → {FINAL_MODEL_DIR}")