# import os
# import torch
# from PIL import Image
# from transformers import (
#     CLIPProcessor,
#     CLIPModel,
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM
# )
# from ultralytics import YOLO
# import numpy as np
# # ─────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────

# FINETUNED_CLIP_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "checkpoint_best.pt"
# )

# TIER1_CONF = 0.30
# TIER2_CONF = 0.22

# device = "cpu"
# torch.set_num_threads(4)

# # ─────────────────────────────────────────────────────────────
# # LABELS
# # ─────────────────────────────────────────────────────────────

# INDIAN_LABELS = [
#     "a photo of indian roti flatbread","a bowl of paneer curry with rich gravy","a plate of steamed basmati rice","a serving of indian dal lentil curry",
#     "a crispy dosa on a plate","a deep fried vada on a plate","a bowl of sambar lentil soup","a bowl of coconut chutney","idli served with chutney and sambar","a plate of indian vegetable sabzi",
#     "soft steamed idli","a plain dosa","a masala dosa filled with potato stuffing","a fried vada snack","a medu vada savory donut","a thick uttapam with toppings",
#     "a plate of biryani rice dish","a plate of chicken biryani","a plate of fried rice","a plate of paneer fried rice",
#     "a plate of vegetable pulao","chole bhature with chickpea curry and poori","pav bhaji mashed vegetable curry served with bread rolls and butter","a crispy samosa snack","a fried aloo tikki patty",
#     "paneer butter masala with visible paneer cubes in creamy tomato gravy","palak paneer with spinach gravy","dal makhani with creamy lentils",
#     "rajma kidney bean curry","aloo gobi with potato and cauliflower","butter chicken in rich tomato gravy","tandoori chicken grilled with spices",
#     "indian roti bread","soft naan bread","stuffed paratha flatbread","deep fried puri bread","fluffy bhatura bread",
#     "gulab jamun soaked in sugar syrup","rasgulla in sugar syrup","crispy jalebi dessert","sweet laddu balls","kheer rice pudding",
#     "sweet halwa dessert","poha flattened rice dish","upma semolina breakfast dish","dhokla steamed savory cake","pani puri street food snack",
#     "a bowl of sambar","a bowl of chutney","a bowl of indian curry","a golden brown puffed deep fried poori bread with crispy texture","a golden brown puffed deep fried poori bread with crispy texture"
# ]
# GLOBAL_LABELS = [
#     "a slice of pizza with toppings","a burger with bun and patty","a plate of pasta with sauce","a sandwich with fillings","a wrap with vegetables or meat",
#     "a bowl of noodles","a plate of fried noodles","sushi rolls on a plate","a bowl of fresh salad","fried chicken pieces","grilled chicken with seasoning",
#     "chicken wings with sauce","a serving of french fries","an omelette made with eggs","scrambled eggs on a plate","a stack of pancakes",
#     "waffles with syrup","tacos with filling","a hot dog in a bun","spring rolls with filling","dumplings on a plate","a steak piece of meat","a bowl of soup","a bowl of ramen noodles",
#     "a scoop of ice cream","a slice of cake","a chocolate cake slice","cookies on a plate","a donut with icing"
# ]
# ALL_LABELS = INDIAN_LABELS + GLOBAL_LABELS

# FOOD_TEXTS = [
#     "a photo of food", "a plate of food",
#     "a delicious meal", "indian food", "street food",
# ]

# NON_FOOD_TEXTS = [
#     "a person", "a car", "a building",
#     "random object", "blank background",
# ]

# # ─────────────────────────────────────────────────────────────
# # MODEL CACHE
# # ─────────────────────────────────────────────────────────────

# _finetuned_clip = None
# _general_clip = None
# _clip_processor = None

# _tokenizer = None
# _text_model = None
# yolo_conf=0.15
# # ─────────────────────────────────────────────────────────────
# # LOADERS
# # ─────────────────────────────────────────────────────────────
# _yolo_model = None

# def _get_yolo():
#     global _yolo_model
#     if _yolo_model is None:
#         print("[Loading YOLOv8]")
#         _yolo_model = YOLO("yolov8n.pt")  # or your trained model
#     return _yolo_model

# def _crop_image(image, box, pad=10):
#     x1, y1, x2, y2 = map(int, box)

#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(image.width, x2 + pad)
#     y2 = min(image.height, y2 + pad)

#     return image.crop((x1, y1, x2, y2))

# def _get_clip_processor():
#     global _clip_processor
#     if _clip_processor is None:
#         _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return _clip_processor


# def _get_finetuned_clip():
#     global _finetuned_clip

#     if _finetuned_clip is None:
#         if os.path.exists(FINETUNED_CLIP_PATH):
#             print("[Loading fine-tuned CLIP properly]")

#             checkpoint = torch.load(FINETUNED_CLIP_PATH, map_location="cpu")

#             # 🔥 KEY FIX
#             state_dict = checkpoint["model"]   # <-- extract actual model

#             model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#             model.load_state_dict(state_dict, strict=False)  # allow mismatch
#             model.eval()

#             _finetuned_clip = model
#         else:
#             _finetuned_clip = "missing"

#     return None if _finetuned_clip == "missing" else _finetuned_clip
# def _get_general_clip():
#     global _general_clip
#     if _general_clip is None:
#         print("[Loading general CLIP]")
#         _general_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         _general_clip.eval()
#     return _general_clip


# def _get_text_model():
#     global _tokenizer, _text_model
#     if _text_model is None:
#         print("[Loading FLAN-T5]")
#         model_name = "google/flan-t5-base"

#         _tokenizer = AutoTokenizer.from_pretrained(model_name)
#         _text_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     return _tokenizer, _text_model

# # ─────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────

# def _make_prompts(labels):
#     prompts = []
#     for lbl in labels:
#         prompts.extend([
#             f"a photo of {lbl}",
#             f"a plate of {lbl}",
#             f"a delicious {lbl}",
#             f"indian food {lbl}",
#         ])
#     return prompts

# def _clip_classify(model, image, labels, top_k=3):
#     processor = _get_clip_processor()
#     prompts = _make_prompts(labels)

#     inputs = processor(
#         text=prompts,
#         images=image,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = outputs.logits_per_image.softmax(dim=1)[0]

#     scores = []
#     for i, lbl in enumerate(labels):
#         score = probs[i*4:(i+1)*4].max().item()
#         scores.append((lbl, score))

#     scores.sort(key=lambda x: x[1], reverse=True)

#     return [l for l, _ in scores[:top_k]], [s for _, s in scores[:top_k]]
# def debug_print_detection(idx, box, yolo_conf, label, clip_conf):
#     print("\n" + "="*50)
#     print(f"[OBJECT {idx}]")
#     print(f"📦 Box: {box}")
#     print(f"🎯 YOLO Confidence: {round(yolo_conf, 3)}")
#     print(f"🍽️ Predicted Food: {label}")
#     print(f"🧠 CLIP Confidence: {round(clip_conf, 3)}")
#     print("="*50)

# def _is_food(image):
#     model = _get_general_clip()
#     processor = _get_clip_processor()

#     texts = FOOD_TEXTS + NON_FOOD_TEXTS
#     inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = outputs.logits_per_image.softmax(dim=1)[0]

#     food_conf = probs[:len(FOOD_TEXTS)].sum().item()
#     non_food_conf = probs[len(FOOD_TEXTS):].sum().item()

#     return food_conf > non_food_conf, round(food_conf, 3)


# def generate_caption(label):
#     tokenizer, model = _get_text_model()

#     # prompt = f"What is {label}? Describe it like a food menu in one sentence."
#     prompt = f"""
#     Describe the food item '{label}' in a simple, structured way like a food dataset.

#     Include:
#     - main ingredients
#     - cooking method
#     - texture or form

# Keep it short, factual, and one sentence only.
# """

#     inputs = tokenizer(prompt, return_tensors="pt")

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=30
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ─────────────────────────────────────────────────────────────
# # MAIN PIPELINE
# # ─────────────────────────────────────────────────────────────

# def process_image(image):
#     image = image.convert("RGB")

#     # ─────────────────────────────
#     # Step 1: Food check
#     # ─────────────────────────────
#     is_food_flag, food_conf = _is_food(image)
#     if not is_food_flag:
#         return {
#             "status": "not_food",
#             "message": "This is not a food image"
#         }

#     # ─────────────────────────────
#     # Step 2: YOLO detection
#     # ─────────────────────────────
#     yolo = _get_yolo()
#     results = yolo(image, conf=0.15)

#     boxes = results[0].boxes
#     valid_boxes = []

#     if boxes is not None:
#         for box_obj in boxes:
#             conf = float(box_obj.conf[0])
#             if conf > 0.4:   # strong detections only
#                 valid_boxes.append(box_obj)

#     num_objects = len(valid_boxes)

#     print("\n" + "#"*60)
#     print(f"[YOLO] Valid objects after filtering: {num_objects}")
#     print("#"*60)

#     # ─────────────────────────────
#     # CASE 1: YOLO unreliable → CLIP multi detection
#     # ─────────────────────────────
#     if num_objects <= 1:

#         print("\n🚨 YOLO unreliable → using CLIP multi-food detection")

#         labels, scores = _clip_classify(_get_general_clip(), image, ALL_LABELS, top_k=3)

#         foods = []
#         for l, s in zip(labels, scores):
#             if s > 0.12:
#                 foods.append({
#                     "label": l,
#                     "confidence": round(s, 3),
#                     "caption": f"a delicious plate of {l}",
#                     "box": [0, 0, image.width, image.height]  # ⚠️ FULL IMAGE fallback
#                 })

#         # remove duplicates
#         unique_foods = []
#         seen = set()
#         for f in foods:
#             key = f["label"].split()[-1]
#             if key not in seen:
#                 unique_foods.append(f)
#                 seen.add(key)

#         foods = sorted(unique_foods, key=lambda x: x["confidence"], reverse=True)

#         return {
#             "status": "food",
#             "primary": foods[0]["label"] if foods else "unknown",
#             "count": len(foods),
#             "foods": foods
#         }

#     # ─────────────────────────────
#     # CASE 2: YOLO reliable → crop + CLIP
#     # ─────────────────────────────
#     else:
#         print("\n✅ YOLO reliable → using multi-object pipeline")

#         results_list = []

#         for i, box_obj in enumerate(valid_boxes):

#             box = box_obj.xyxy[0].tolist()  # ✅ KEEP THIS
#             yolo_conf = float(box_obj.conf[0])

#             crop = _crop_image(image, box)

#             # Try fine-tuned CLIP
#             finetuned = _get_finetuned_clip()
#             if finetuned is not None:
#                 labels, scores = _clip_classify(finetuned, crop, ALL_LABELS)
#                 label, clip_conf = labels[0], scores[0]

#                 debug_print_detection(i+1, box, yolo_conf, label, clip_conf)

#                 if clip_conf >= TIER1_CONF:
#                     results_list.append({
#                         "food": label,
#                         "confidence": round(clip_conf, 3),
#                         "box": box   # ✅ IMPORTANT
#                     })
#                     continue

#             # fallback to general CLIP
#             labels, scores = _clip_classify(_get_general_clip(), crop, ALL_LABELS)
#             label, clip_conf = labels[0], scores[0]

#             debug_print_detection(i+1, box, yolo_conf, label, clip_conf)

#             results_list.append({
#                 "food": label,
#                 "confidence": round(clip_conf, 3),
#                 "box": box   # ✅ IMPORTANT
#             })

#         return {
#             "status": "food",
#             "primary": results_list[0]["food"] if results_list else "unknown",
#             "count": len(results_list),
#             "foods": [
#                 {
#                     "label": item["food"],
#                     "confidence": item["confidence"],
#                     "caption": f"a delicious plate of {item['food']}",
#                     "box": item["box"]   # ✅ PASS TO BACKEND
#                 }
#                 for item in results_list
#             ]
#         }
    

# import os
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
# from ultralytics import YOLO

# # ─────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────

# FINETUNED_CLIP_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "checkpoint_best.pt"
# )

# TIER1_CONF        = 0.22   # fine-tuned CLIP cosine similarity threshold
# TIER2_CONF        = 0.18   # general CLIP cosine similarity threshold
# MIN_REPORT_CONF   = 0.16   # don't report detections below this (multi-object)
# MIN_SINGLE_CONF   = 0.22   # stricter threshold for single-item (Case 1)
# TOP_GAP_THRESHOLD = 0.04   # if rank-1 leads rank-2 by this much → single item only
# YOLO_CONF         = 0.15   # YOLO detection confidence
# YOLO_VALID_CONF   = 0.40   # YOLO box must exceed this to be trusted

# device = "cpu"
# torch.set_num_threads(4)

# # ─────────────────────────────────────────────────────────────
# # LABELS — organized by category for two-stage classification
# # ─────────────────────────────────────────────────────────────

# CATEGORIES = {
#     "indian_bread": [
#         "roti", "naan", "paratha", "puri", "bhatura",
#         "chapati", "kulcha", "lachha paratha", "missi roti",
#     ],
#     "indian_curry_paneer": [
#         "paneer butter masala", "palak paneer", "kadai paneer",
#         "shahi paneer", "malai kofta", "paneer tikka masala",
#     ],
#     "indian_curry_dal": [
#         "dal", "dal makhani", "rajma", "chole", "kadhi",
#         "moong dal", "chana dal", "sambar",
#     ],
#     "indian_curry_veg": [
#         "aloo gobi", "baingan bharta", "vegetable curry",
#         "aloo matar", "jeera aloo", "matar paneer",
#         "mixed vegetable curry",
#     ],
#     "indian_curry_chicken": [
#         "chicken curry", "butter chicken", "chicken korma",
#         "chicken chettinad", "chicken tikka masala", "kadai chicken",
#         "chicken do pyaza",
#     ],
#     "indian_curry_meat": [
#         "mutton curry", "keema", "lamb curry",
#         "mutton rogan josh", "nihari",
#     ],
#     "indian_curry_seafood": [
#         "fish curry", "fish fry", "prawn curry",
#         "prawn fry", "egg curry", "egg bhurji",
#     ],
#     "tandoor_grill": [
#         "tandoori chicken", "chicken tikka", "seekh kebab",
#         "paneer tikka", "fish tikka", "tandoori roti", "reshmi kebab",
#     ],
#     "rice_dishes": [
#         "biryani", "chicken biryani", "mutton biryani", "veg biryani",
#         "veg pulao", "fried rice", "jeera rice", "curd rice",
#         "lemon rice", "tamarind rice", "ghee rice", "khichdi",
#     ],
#     "south_indian": [
#         "idli", "dosa", "masala dosa", "rava dosa", "pesarattu",
#         "uttapam", "medu vada", "vada", "pongal", "upma",
#         "poha", "appam",
#     ],
#     "indian_snacks_street": [
#         "samosa", "aloo tikki", "pani puri", "bhel puri",
#         "sev puri", "dahi puri", "pav bhaji", "chole bhature",
#         "bread pakora", "sabudana khichdi", "kathi roll", "paneer roll",
#     ],
#     "indian_sweets": [
#         "gulab jamun", "rasgulla", "jalebi", "laddu", "kheer",
#         "halwa", "barfi", "peda", "ras malai", "shrikhand",
#         "soan papdi", "mysore pak", "gajar halwa",
#     ],
#     "indian_beverages": [
#         "masala tea", "chai", "lassi", "buttermilk",
#         "falooda", "rose milk", "badam milk", "jaljeera",
#         "sugarcane juice", "nimbu pani",
#     ],
#     "western_pizza_pasta": [
#         "pizza", "pasta", "spaghetti bolognese", "penne alfredo",
#         "lasagna", "mac and cheese", "ravioli", "carbonara", "gnocchi",
#     ],
#     "western_burgers_sandwiches": [
#         "burger", "cheeseburger", "sandwich", "wrap",
#         "hot dog", "tacos", "nachos", "corn dog", "club sandwich",
#     ],
#     "western_chicken": [
#         "fried chicken", "grilled chicken", "chicken wings",
#         "chicken nuggets", "chicken strips", "bbq chicken",
#     ],
#     "western_mains": [
#         "steak", "grilled steak", "bbq ribs", "beef stew",
#         "meatloaf", "lamb chops", "roast chicken",
#         "turkey roast", "fish and chips", "salmon fillet",
#     ],
#     "western_sides": [
#         "french fries", "loaded fries", "onion rings",
#         "coleslaw", "mashed potato",
#     ],
#     "asian_noodles": [
#         "noodles", "fried noodles", "ramen", "pad thai",
#         "dumplings", "spring rolls", "dim sum", "wonton soup",
#     ],
#     "breakfast": [
#         "omelette", "scrambled eggs", "fried eggs", "poached eggs",
#         "eggs benedict", "bacon", "toast", "bagel", "croissant",
#         "pancakes", "waffles", "granola", "avocado toast",
#     ],
#     "salads": [
#         "salad", "caesar salad", "greek salad", "coleslaw",
#         "quinoa salad", "fruit salad", "vegetable salad",
#     ],
#     "desserts_baked": [
#         "cake", "chocolate cake", "vanilla cake", "red velvet cake",
#         "black forest cake", "carrot cake", "cheesecake",
#         "brownie", "cupcake", "muffin", "apple pie",
#         "blueberry pie", "chocolate mousse", "pastry", "eclair",
#     ],
#     "desserts_frozen": [
#         "ice cream", "ice cream sundae", "gelato", "frozen yogurt",
#     ],
#     "beverages_cold": [
#         "milkshake", "smoothie", "juice", "orange juice",
#         "mango juice", "lemonade", "iced tea", "cold coffee",
#         "protein shake", "bubble tea",
#     ],
#     "fruits": [
#         "apple", "banana", "orange", "mango", "grapes",
#         "pineapple", "watermelon", "papaya", "strawberry",
#         "blueberry", "kiwi", "peach", "pomegranate",
#     ],
#     "vegetables": [
#         "tomato", "onion", "potato", "carrot", "cucumber",
#         "capsicum", "bell pepper", "broccoli", "cauliflower",
#         "cabbage", "spinach", "lettuce", "peas", "corn",
#         "eggplant", "zucchini", "mushroom", "garlic", "ginger",
#     ],
# }

# ALL_LABELS = [label for labels in CATEGORIES.values() for label in labels]

# CATEGORY_DISPLAY = {
#     "indian_bread":               "Indian flatbread like roti or naan",
#     "indian_curry_paneer":        "Indian paneer curry dish",
#     "indian_curry_dal":           "Indian dal or lentil dish",
#     "indian_curry_veg":           "Indian vegetable curry",
#     "indian_curry_chicken":       "Indian chicken curry",
#     "indian_curry_meat":          "Indian mutton or meat curry",
#     "indian_curry_seafood":       "Indian seafood or egg curry",
#     "tandoor_grill":              "tandoor grilled Indian food",
#     "rice_dishes":                "rice dish like biryani or pulao",
#     "south_indian":               "South Indian food like dosa or idli",
#     "indian_snacks_street":       "Indian street food or snack",
#     "indian_sweets":              "Indian sweet or dessert",
#     "indian_beverages":           "Indian beverage like chai or lassi",
#     "western_pizza_pasta":        "pizza or pasta dish",
#     "western_burgers_sandwiches": "burger or sandwich",
#     "western_chicken":            "Western fried or grilled chicken",
#     "western_mains":              "Western main course like steak or roast",
#     "western_sides":              "Western side dish like fries or rings",
#     "asian_noodles":              "Asian noodles or dumplings",
#     "breakfast":                  "breakfast food like eggs or pancakes",
#     "salads":                     "fresh salad",
#     "desserts_baked":             "baked dessert like cake or brownie",
#     "desserts_frozen":            "frozen dessert like ice cream",
#     "beverages_cold":             "cold beverage or drink",
#     "fruits":                     "fresh fruit",
#     "vegetables":                 "raw or fresh vegetable",
# }

# # ─────────────────────────────────────────────────────────────
# # FOOD / NON-FOOD GATE TEXTS
# # ─────────────────────────────────────────────────────────────

# FOOD_TEXTS = [
#     "a photo of food",
#     "a plate of food",
#     "a delicious meal",
#     "indian food on a plate",
#     "street food being served",
# ]

# NON_FOOD_TEXTS = [
#     "a photo of a person",
#     "a photo of a car",
#     "a photo of a building",
#     "a random object",
#     "a blank background",
# ]

# # ─────────────────────────────────────────────────────────────
# # MODEL CACHE
# # ─────────────────────────────────────────────────────────────

# _finetuned_clip = None
# _general_clip   = None
# _clip_processor = None
# _tokenizer      = None
# _text_model     = None
# _yolo_model     = None

# # ─────────────────────────────────────────────────────────────
# # LOADERS
# # ─────────────────────────────────────────────────────────────

# def _get_clip_processor():
#     global _clip_processor
#     if _clip_processor is None:
#         print("[Loading CLIP processor]")
#         _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return _clip_processor


# def _get_general_clip():
#     global _general_clip
#     if _general_clip is None:
#         print("[Loading general CLIP]")
#         _general_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         _general_clip.eval()
#     return _general_clip


# def _get_finetuned_clip():
#     global _finetuned_clip
#     if _finetuned_clip is None:
#         if os.path.exists(FINETUNED_CLIP_PATH):
#             print("[Loading fine-tuned CLIP]")
#             checkpoint = torch.load(FINETUNED_CLIP_PATH, map_location="cpu")
#             state_dict = checkpoint["model"]
#             model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#             model.load_state_dict(state_dict, strict=False)
#             model.eval()
#             _finetuned_clip = model
#         else:
#             print("[WARNING] Fine-tuned CLIP checkpoint not found, using general CLIP only")
#             _finetuned_clip = "missing"
#     return None if _finetuned_clip == "missing" else _finetuned_clip


# def _get_yolo():
#     global _yolo_model
#     if _yolo_model is None:
#         print("[Loading YOLOv8]")
#         _yolo_model = YOLO("yolov8n.pt")
#     return _yolo_model


# def _get_text_model():
#     global _tokenizer, _text_model
#     if _text_model is None:
#         print("[Loading FLAN-T5]")
#         model_name  = "google/flan-t5-base"
#         _tokenizer  = AutoTokenizer.from_pretrained(model_name)
#         _text_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return _tokenizer, _text_model

# # ─────────────────────────────────────────────────────────────
# # PROMPT BUILDER — context-aware, 5 prompts per label
# # ─────────────────────────────────────────────────────────────

# _RAW_LABELS       = set(CATEGORIES["fruits"] + CATEGORIES["vegetables"])
# PROMPTS_PER_LABEL = 5   # must match _make_prompts output length

# def _make_prompts(label: str) -> list:
#     """
#     5 prompts per label.
#     Raw produce gets fresh/raw prompts; cooked food gets serving prompts.
#     Avoids semantically wrong combos like 'street food apple'.
#     """
#     if label in _RAW_LABELS:
#         return [
#             f"a photo of {label}",
#             f"a fresh {label}",
#             f"raw {label}",
#             f"a close-up of {label}",
#             f"{label} on a white background",
#         ]
#     else:
#         return [
#             f"a photo of {label}",
#             f"a plate of {label}",
#             f"a close-up of {label}",
#             f"a delicious serving of {label}",
#             f"freshly made {label}",
#         ]

# # ─────────────────────────────────────────────────────────────
# # CORE: COSINE-SIMILARITY CLIP CLASSIFY
# # ─────────────────────────────────────────────────────────────

# def _clip_classify(model, image, labels: list, top_k: int = 3) -> tuple:
#     """
#     Classify image against labels using cosine similarity (NOT softmax).
#     Aggregates over PROMPTS_PER_LABEL prompts per label by taking the max.

#     Returns:
#         top_labels  — list of str, length top_k
#         top_scores  — list of float, cosine similarity in [~0.10, ~0.40]
#     """
#     processor   = _get_clip_processor()
#     all_prompts = []
#     for lbl in labels:
#         all_prompts.extend(_make_prompts(lbl))

#     inputs = processor(
#         text=all_prompts,
#         images=image,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#     )

#     with torch.no_grad():
#         image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
#         text_features  = model.get_text_features(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#         )

#     # L2-normalize both
#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

#     # Cosine similarities: shape (num_prompts,)
#     similarities = (image_features @ text_features.T)[0]

#     # Aggregate per label: max over its PROMPTS_PER_LABEL prompts
#     scores = []
#     for i, lbl in enumerate(labels):
#         start = i * PROMPTS_PER_LABEL
#         end   = start + PROMPTS_PER_LABEL
#         score = similarities[start:end].max().item()
#         scores.append((lbl, score))

#     scores.sort(key=lambda x: x[1], reverse=True)
#     top = scores[:top_k]
#     return [l for l, _ in top], [s for _, s in top]

# # ─────────────────────────────────────────────────────────────
# # TWO-STAGE CLASSIFICATION: coarse → fine
# # ─────────────────────────────────────────────────────────────

# def _clip_classify_two_stage(model, image, top_k: int = 3) -> tuple:
#     """
#     Stage 1 — pick top 2 coarse categories using short descriptive prompts.
#     Stage 2 — fine-grained cosine-sim classify only within those candidates.
#     Falls back to ALL_LABELS if no category clears the minimum threshold.
#     """
#     processor  = _get_clip_processor()
#     cat_names  = list(CATEGORIES.keys())
#     cat_texts  = [f"a photo of {CATEGORY_DISPLAY[c]}" for c in cat_names]

#     inputs = processor(
#         text=cat_texts,
#         images=image,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#     )

#     with torch.no_grad():
#         img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
#         txt_feat = model.get_text_features(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#         )

#     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
#     txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
#     cat_sims = (img_feat @ txt_feat.T)[0]

#     top2_idx    = cat_sims.topk(2).indices.tolist()
#     top2_scores = [cat_sims[i].item() for i in top2_idx]

#     COARSE_MIN = 0.15
#     if top2_scores[0] < COARSE_MIN:
#         print(f"[Two-stage] Coarse sim too low ({top2_scores[0]:.3f}), "
#               f"falling back to ALL_LABELS")
#         return _clip_classify(model, image, ALL_LABELS, top_k=top_k)

#     candidate_labels = []
#     for idx in top2_idx:
#         candidate_labels.extend(CATEGORIES[cat_names[idx]])

#     print(f"[Two-stage] Top categories : {[cat_names[i] for i in top2_idx]}")
#     print(f"[Two-stage] Cat sims       : {[round(s, 3) for s in top2_scores]}")
#     print(f"[Two-stage] Candidates     : {candidate_labels}")

#     return _clip_classify(model, image, candidate_labels, top_k=top_k)

# # ─────────────────────────────────────────────────────────────
# # FOOD / NON-FOOD GATE
# # ─────────────────────────────────────────────────────────────

# def _is_food(image) -> tuple:
#     """
#     Returns (is_food: bool, food_confidence: float).
#     Uses mean cosine similarity so score is not diluted by label count.
#     """
#     model     = _get_general_clip()
#     processor = _get_clip_processor()

#     all_texts = FOOD_TEXTS + NON_FOOD_TEXTS
#     inputs    = processor(
#         text=all_texts, images=image,
#         return_tensors="pt", padding=True,
#     )

#     with torch.no_grad():
#         img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
#         txt_feat = model.get_text_features(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#         )

#     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
#     txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
#     sims     = (img_feat @ txt_feat.T)[0]

#     food_conf     = sims[:len(FOOD_TEXTS)].mean().item()
#     non_food_conf = sims[len(FOOD_TEXTS):].mean().item()

#     return food_conf > non_food_conf, round(food_conf, 3)

# # ─────────────────────────────────────────────────────────────
# # IMAGE HELPERS
# # ─────────────────────────────────────────────────────────────

# def _crop_image(image, box, pad: int = 10):
#     x1, y1, x2, y2 = map(int, box)
#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(image.width,  x2 + pad)
#     y2 = min(image.height, y2 + pad)
#     return image.crop((x1, y1, x2, y2))

# # ─────────────────────────────────────────────────────────────
# # CAPTION GENERATOR
# # ─────────────────────────────────────────────────────────────

# def generate_caption(label: str) -> str:
#     tokenizer, model = _get_text_model()
#     prompt = (
#         f"Describe the food item '{label}' in a simple, structured way like a food dataset. "
#         f"Include: main ingredients, cooking method, texture or form. "
#         f"Keep it short, factual, and one sentence only."
#     )
#     inputs  = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=30)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ─────────────────────────────────────────────────────────────
# # DEBUG HELPER
# # ─────────────────────────────────────────────────────────────

# def _debug_detection(idx, box, yolo_conf, label, clip_conf):
#     print("\n" + "=" * 50)
#     print(f"[OBJECT {idx}]")
#     print(f"  Box           : {[round(v, 1) for v in box]}")
#     print(f"  YOLO conf     : {round(yolo_conf, 3)}")
#     print(f"  Predicted food: {label}")
#     print(f"  CLIP cos-sim  : {round(clip_conf, 3)}")
#     print("=" * 50)

# # ─────────────────────────────────────────────────────────────
# # CASE 1 HELPER — smart deduplication for single-plate images
# # ─────────────────────────────────────────────────────────────

# def _is_duplicate(new_label: str, accepted_labels: set) -> bool:
#     """
#     Returns True if new_label shares ANY word with an already-accepted label.
#     Catches variants like:
#       'penne alfredo' vs 'pasta'        → share nothing → NOT caught here
#       'penne alfredo' vs 'penne'        → share 'penne' → caught
#       'spaghetti bolognese' vs 'pasta'  → share nothing → NOT caught
#     Combined with gap filter, covers most false-multi cases.
#     """
#     new_words = set(new_label.lower().split())
#     for seen_lbl in accepted_labels:
#         seen_words = set(seen_lbl.lower().split())
#         if new_words & seen_words:
#             print(f"  [Dedup] Skipping '{new_label}' (overlaps with '{seen_lbl}')")
#             return True
#     return False


# def _build_foods_case1(labels: list, scores: list, image_w: int, image_h: int) -> list:
#     """
#     Filters and deduplicates CLIP results for single-plate (Case 1) images.

#     Rules applied in order:
#       1. Skip if score < MIN_SINGLE_CONF
#       2. Stop if gap between rank-1 and current > TOP_GAP_THRESHOLD
#          (current item is much weaker than top → single dominant food)
#       3. Skip if label shares a word with any already-accepted label
#       4. Hard cap at 3 items
#     """
#     foods = []
#     seen  = set()

#     for idx, (lbl, score) in enumerate(zip(labels, scores)):

#         # Rule 1: minimum confidence
#         if score < MIN_SINGLE_CONF:
#             print(f"  [Conf filter] '{lbl}' score {score:.3f} < {MIN_SINGLE_CONF}, stopping")
#             break

#         # Rule 2: gap filter — top item has a clear lead
#         if idx > 0 and (scores[0] - score) > TOP_GAP_THRESHOLD:
#             print(f"  [Gap filter] Stopping at '{lbl}': "
#                   f"gap={scores[0] - score:.3f} > {TOP_GAP_THRESHOLD}")
#             break

#         # Rule 3: word-overlap deduplication
#         if _is_duplicate(lbl, seen):
#             continue

#         seen.add(lbl)
#         foods.append({
#             "label":      lbl,
#             "confidence": round(score, 3),
#             "caption":    f"a delicious plate of {lbl}",
#             "box":        [0, 0, image_w, image_h],
#         })

#         # Rule 4: hard cap
#         if len(foods) >= 3:
#             print("  [Hard cap] Reached 3 items, stopping")
#             break

#     return foods

# # ─────────────────────────────────────────────────────────────
# # MAIN PIPELINE
# # ─────────────────────────────────────────────────────────────

# def process_image(image: Image.Image) -> dict:
#     """
#     Full pipeline:
#       1. Food / non-food gate
#       2. YOLO object detection
#       3a. YOLO unreliable (<=1 box) → two-stage CLIP on full image
#       3b. YOLO reliable  (>1 boxes) → crop each box, two-stage CLIP per crop

#     Returns:
#       {
#         status  : "food" | "not_food",
#         primary : str,
#         count   : int,
#         foods   : [ {label, confidence, caption, box}, ... ]
#       }
#     """
#     image = image.convert("RGB")

#     # ── Step 1: Food gate ────────────────────────────────────
#     is_food_flag, food_conf = _is_food(image)
#     print(f"\n[Food gate] is_food={is_food_flag}  food_conf={food_conf}")

#     if not is_food_flag:
#         return {
#             "status":  "not_food",
#             "message": "This is not a food image",
#         }

#     # ── Step 2: YOLO detection ───────────────────────────────
#     yolo        = _get_yolo()
#     results     = yolo(image, conf=YOLO_CONF)
#     boxes       = results[0].boxes
#     valid_boxes = []

#     if boxes is not None:
#         for box_obj in boxes:
#             conf = float(box_obj.conf[0])
#             if conf > YOLO_VALID_CONF:
#                 valid_boxes.append(box_obj)

#     num_objects = len(valid_boxes)
#     print(f"\n{'#'*60}")
#     print(f"[YOLO] Valid boxes (conf > {YOLO_VALID_CONF}): {num_objects}")
#     print(f"{'#'*60}\n")

#     # ── Case 1: YOLO unreliable → full-image two-stage CLIP ──
#     if num_objects <= 1:
#         print("🚨 YOLO unreliable → two-stage CLIP on full image")

#         model          = _get_finetuned_clip() or _get_general_clip()
#         labels, scores = _clip_classify_two_stage(model, image, top_k=5)

#         print(f"\n[Case 1] Raw CLIP results:")
#         for lbl, sc in zip(labels, scores):
#             print(f"  {lbl:30s}  {sc:.3f}")

#         foods = _build_foods_case1(labels, scores, image.width, image.height)
#         foods.sort(key=lambda x: x["confidence"], reverse=True)

#         print(f"\n[Case 1] Final detections: {[f['label'] for f in foods]}")

#         return {
#             "status":  "food",
#             "primary": foods[0]["label"] if foods else "unknown",
#             "count":   len(foods),
#             "foods":   foods,
#         }

#     # ── Case 2: YOLO reliable → crop + two-stage CLIP ────────
#     print("✅ YOLO reliable → multi-object pipeline")

#     results_list = []
#     finetuned    = _get_finetuned_clip()

#     for i, box_obj in enumerate(valid_boxes):
#         box       = box_obj.xyxy[0].tolist()
#         yolo_conf = float(box_obj.conf[0])
#         crop      = _crop_image(image, box)

#         # Try fine-tuned CLIP first
#         if finetuned is not None:
#             labels, scores = _clip_classify_two_stage(finetuned, crop, top_k=3)
#             label, clip_conf = labels[0], scores[0]
#             _debug_detection(i + 1, box, yolo_conf, label, clip_conf)

#             if clip_conf >= TIER1_CONF:
#                 results_list.append({
#                     "food":       label,
#                     "confidence": round(clip_conf, 3),
#                     "box":        box,
#                 })
#                 continue

#         # Fallback: general CLIP
#         labels, scores = _clip_classify_two_stage(_get_general_clip(), crop, top_k=3)
#         label, clip_conf = labels[0], scores[0]
#         _debug_detection(i + 1, box, yolo_conf, label, clip_conf)

#         if clip_conf >= TIER2_CONF:
#             results_list.append({
#                 "food":       label,
#                 "confidence": round(clip_conf, 3),
#                 "box":        box,
#             })
#         else:
#             print(f"  ⚠️  Skipping object {i+1}: "
#                   f"clip_conf {clip_conf:.3f} < TIER2_CONF {TIER2_CONF}")

#     return {
#         "status":  "food",
#         "primary": results_list[0]["food"] if results_list else "unknown",
#         "count":   len(results_list),
#         "foods": [
#             {
#                 "label":      item["food"],
#                 "confidence": item["confidence"],
#                 "caption":    f"a delicious plate of {item['food']}",
#                 "box":        item["box"],
#             }
#             for item in results_list
#         ],
#     }

# import os
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
# from ultralytics import YOLO

# # ─────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────

# FINETUNED_CLIP_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "checkpoint_best.pt"
# )

# TIER1_CONF        = 0.22   # fine-tuned CLIP cosine similarity threshold
# TIER2_CONF        = 0.18   # general CLIP cosine similarity threshold
# MIN_REPORT_CONF   = 0.16   # don't report detections below this (multi-object)
# MIN_SINGLE_CONF   = 0.22   # stricter threshold for single-item (Case 1)
# TOP_GAP_THRESHOLD = 0.06   # if rank-1 leads rank-2 by this much → single item only
# YOLO_CONF         = 0.15   # YOLO detection confidence
# YOLO_VALID_CONF   = 0.40   # YOLO box must exceed this to be trusted

# device = "cpu"
# torch.set_num_threads(4)

# # ─────────────────────────────────────────────────────────────
# # LABELS — organized by category for two-stage classification
# # ASA24-aligned: covers Indian, Western, Asian, Latin, Middle
# # Eastern, dairy, grains, snacks, beverages, condiments, etc.
# # ─────────────────────────────────────────────────────────────

# CATEGORIES = {
#     # ── INDIAN ───────────────────────────────────────────────
#     "indian_bread": [
#         "roti", "naan", "paratha", "puri", "bhatura",
#         "chapati", "kulcha", "lachha paratha", "missi roti",
#         "thepla", "phulka", "bajra roti", "makki di roti",
#         "besan cheela",
#     ],
#     "indian_curry_paneer": [
#         "paneer butter masala", "palak paneer", "kadai paneer",
#         "shahi paneer", "malai kofta", "paneer tikka masala",
#         "paneer bhurji", "chilli paneer",
#     ],
#     "indian_curry_dal": [
#         "dal", "dal makhani", "rajma", "chole", "kadhi",
#         "moong dal", "chana dal", "sambar", "dal tadka",
#         "dal fry", "masoor dal", "lobiya curry",
#     ],
#     "indian_curry_veg": [
#         "aloo gobi", "baingan bharta", "vegetable curry",
#         "aloo matar", "jeera aloo", "matar paneer",
#         "mixed vegetable curry", "bhindi masala", "lauki sabzi",
#         "tinda sabzi", "arbi curry", "capsicum masala",
#         "stuffed capsicum", "undhiyu",
#     ],
#     "indian_curry_chicken": [
#         "chicken curry", "butter chicken", "chicken korma",
#         "chicken chettinad", "chicken tikka masala", "kadai chicken",
#         "chicken do pyaza", "chicken saag", "chicken rezala",
#         "murgh makhani",
#     ],
#     "indian_curry_meat": [
#         "mutton curry", "keema", "lamb curry",
#         "mutton rogan josh", "nihari", "paya soup",
#         "keema matar", "mutton kalia",
#     ],
#     "indian_curry_seafood": [
#         "fish curry", "fish fry", "prawn curry",
#         "prawn fry", "egg curry", "egg bhurji",
#         "goan fish curry", "kerala fish curry", "crab curry",
#         "fish masala",
#     ],
#     "tandoor_grill": [
#         "tandoori chicken", "chicken tikka", "seekh kebab",
#         "paneer tikka", "fish tikka", "tandoori roti", "reshmi kebab",
#         "boti kebab", "galouti kebab", "hariyali kebab",
#         "shami kebab", "dahi kebab",
#     ],
#     "rice_dishes": [
#         "biryani", "chicken biryani", "mutton biryani", "veg biryani",
#         "veg pulao", "fried rice", "jeera rice", "curd rice",
#         "lemon rice", "tamarind rice", "ghee rice", "khichdi",
#         "egg fried rice", "chicken fried rice", "tehri",
#         "mujaddara",
#     ],
#     "south_indian": [
#         "idli", "dosa", "masala dosa", "rava dosa", "pesarattu",
#         "uttapam", "medu vada", "vada", "pongal", "upma",
#         "poha", "appam", "set dosa", "neer dosa", "bisi bele bath",
#         "rasam", "avial", "kootu", "olan",
#     ],
#     "indian_snacks_street": [
#         "samosa", "aloo tikki", "pani puri", "bhel puri",
#         "sev puri", "dahi puri", "pav bhaji", "chole bhature",
#         "bread pakora", "sabudana khichdi", "kathi roll", "paneer roll",
#         "puri chole", "dabeli", "vada pav", "misal pav",
#         "papdi chaat", "raj kachori", "dhokla", "khandvi",
#         "chakli", "murukku", "mathri", "namak pare",
#         "corn chaat", "aloo chaat",
#     ],
#     "indian_sweets": [
#         "gulab jamun", "rasgulla", "jalebi", "laddu", "kheer",
#         "halwa", "barfi", "peda", "ras malai", "shrikhand",
#         "soan papdi", "mysore pak", "gajar halwa",
#         "besan laddu", "motichoor laddu", "malpua", "imarti",
#         "balushahi", "kalakand", "chum chum", "sandesh",
#         "modak", "puran poli", "ghevar",
#     ],
#     "indian_beverages": [
#         "masala tea", "chai", "lassi", "buttermilk",
#         "falooda", "rose milk", "badam milk", "jaljeera",
#         "sugarcane juice", "nimbu pani", "aam panna",
#         "thandai", "kesar milk", "filter coffee", "bael sherbet",
#     ],

#     # ── WESTERN MAINS ────────────────────────────────────────
#     "western_pizza_pasta": [
#         "pizza", "pasta", "spaghetti bolognese", "penne alfredo",
#         "lasagna", "mac and cheese", "ravioli", "carbonara", "gnocchi",
#         "fettuccine", "linguine", "rigatoni", "tortellini",
#         "pizza margherita", "pepperoni pizza",
#     ],
#     "western_burgers_sandwiches": [
#         "burger", "cheeseburger", "sandwich", "wrap",
#         "hot dog", "tacos", "nachos", "corn dog", "club sandwich",
#         "BLT sandwich", "philly cheesesteak", "sloppy joe",
#         "monte cristo sandwich", "grilled cheese sandwich",
#         "panini", "sub sandwich", "gyro",
#     ],
#     "western_chicken": [
#         "fried chicken", "grilled chicken", "chicken wings",
#         "chicken nuggets", "chicken strips", "bbq chicken",
#         "chicken parmesan", "chicken piccata", "chicken marsala",
#         "chicken pot pie", "buffalo wings",
#     ],
#     "western_mains": [
#         "steak", "grilled steak", "bbq ribs", "beef stew",
#         "meatloaf", "lamb chops", "roast chicken",
#         "turkey roast", "fish and chips", "salmon fillet",
#         "pork chops", "beef pot roast", "veal cutlet",
#         "beef stroganoff", "shepherd's pie", "corned beef",
#         "prime rib", "rack of lamb",
#     ],
#     "western_sides": [
#         "french fries", "loaded fries", "onion rings",
#         "coleslaw", "mashed potato", "baked potato",
#         "potato wedges", "hash browns", "tater tots",
#         "macaroni salad", "potato salad", "corn on the cob",
#         "dinner roll", "garlic bread",
#     ],

#     # ── SOUPS & STEWS ────────────────────────────────────────
#     "soups_stews": [
#         "tomato soup", "chicken noodle soup", "minestrone soup",
#         "french onion soup", "clam chowder", "cream of mushroom soup",
#         "lentil soup", "vegetable soup", "beef stew",
#         "chicken stew", "pho", "miso soup", "hot and sour soup",
#         "mulligatawny soup", "corn soup", "pumpkin soup",
#         "borscht", "gazpacho", "wonton soup", "ramen broth",
#         "dal soup", "rasam",
#     ],

#     # ── ASIAN ────────────────────────────────────────────────
#     "asian_noodles": [
#         "noodles", "fried noodles", "ramen", "pad thai",
#         "dumplings", "spring rolls", "dim sum", "wonton soup",
#         "lo mein", "chow mein", "udon noodles", "soba noodles",
#         "glass noodles", "rice noodles", "laksa", "bibimbap",
#         "banh mi", "pho noodles", "japchae", "chicken noodle",
#     ],
#     "asian_mains": [
#         "sushi", "sashimi", "tempura", "teriyaki chicken",
#         "miso glazed salmon", "kung pao chicken",
#         "general tso chicken", "sweet and sour pork",
#         "mapo tofu", "peking duck", "beef with broccoli",
#         "mongolian beef", "orange chicken", "fried tofu",
#         "pad see ew", "tom yum", "green curry", "massaman curry",
#         "tikka masala", "adobo", "rendang",
#     ],

#     # ── MEXICAN & LATIN ──────────────────────────────────────
#     "mexican_latin": [
#         "tacos", "burritos", "quesadilla", "enchiladas",
#         "tamales", "tostadas", "fajitas", "nachos",
#         "guacamole", "salsa", "chili con carne",
#         "chilaquiles", "pozole", "arroz con pollo",
#         "empanadas", "ceviche", "plantains", "black bean soup",
#         "refried beans", "churros", "tres leches cake",
#     ],

#     # ── MIDDLE EASTERN & MEDITERRANEAN ──────────────────────
#     "middle_eastern_mediterranean": [
#         "hummus", "falafel", "shawarma", "kebab",
#         "pita bread", "tabbouleh", "fattoush salad",
#         "baba ganoush", "dolma", "shakshuka",
#         "moussaka", "spanakopita", "tzatziki",
#         "labneh", "kibbeh", "fatayer",
#         "mansaf", "maqluba", "koshari", "ful medames",
#         "baklava", "kunafa",
#     ],

#     # ── BREAKFAST ────────────────────────────────────────────
#     "breakfast": [
#         "omelette", "scrambled eggs", "fried eggs", "poached eggs",
#         "eggs benedict", "bacon", "toast", "bagel", "croissant",
#         "pancakes", "waffles", "granola", "avocado toast",
#         "breakfast burrito", "french toast", "crepes",
#         "breakfast sandwich", "shakshuka", "porridge", "oatmeal",
#         "cereal with milk", "yogurt parfait", "acai bowl",
#         "frittata", "quiche", "hash and eggs",
#     ],

#     # ── DAIRY & EGGS ─────────────────────────────────────────
#     "dairy_eggs": [
#         "yogurt", "greek yogurt", "cottage cheese",
#         "cheese slice", "cheese block", "cheddar cheese",
#         "mozzarella cheese", "parmesan cheese", "brie",
#         "cream cheese", "ricotta", "hard boiled eggs",
#         "deviled eggs", "milk glass", "butter",
#         "sour cream", "whipped cream", "clotted cream",
#     ],

#     # ── BREADS & GRAINS (WESTERN) ────────────────────────────
#     "breads_grains_western": [
#         "white bread", "whole wheat bread", "sourdough bread",
#         "multigrain bread", "rye bread", "baguette",
#         "dinner roll", "brioche", "focaccia", "ciabatta",
#         "pita bread", "tortilla", "flour tortilla",
#         "corn tortilla", "english muffin", "flatbread",
#         "pretzel", "breadsticks", "biscuit",
#     ],

#     # ── SALADS ───────────────────────────────────────────────
#     "salads": [
#         "salad", "caesar salad", "greek salad", "coleslaw",
#         "quinoa salad", "fruit salad", "vegetable salad",
#         "caprese salad", "nicoise salad", "waldorf salad",
#         "pasta salad", "cobb salad", "spinach salad",
#         "arugula salad", "kale salad", "bean salad",
#         "taco salad", "wedge salad",
#     ],

#     # ── SNACKS & PROCESSED ───────────────────────────────────
#     "snacks_processed": [
#         "potato chips", "tortilla chips", "crackers",
#         "popcorn", "pretzels", "trail mix", "granola bar",
#         "protein bar", "rice cakes", "peanut butter on toast",
#         "cheese and crackers", "vegetable chips",
#         "pork rinds", "beef jerky", "nuts and dried fruit",
#         "hummus and pita chips", "guacamole and chips",
#     ],

#     # ── NUTS & SEEDS ─────────────────────────────────────────
#     "nuts_seeds": [
#         "almonds", "cashews", "walnuts", "peanuts", "pistachios",
#         "pecans", "macadamia nuts", "hazelnuts",
#         "sunflower seeds", "pumpkin seeds", "chia seeds",
#         "flax seeds", "sesame seeds", "mixed nuts",
#         "peanut butter", "almond butter", "tahini",
#     ],

#     # ── LEGUMES & BEANS ──────────────────────────────────────
#     "legumes_beans": [
#         "black beans", "kidney beans", "chickpeas",
#         "lentils", "edamame", "green peas",
#         "navy beans", "pinto beans", "white beans",
#         "split peas", "mung beans", "fava beans",
#         "tofu", "tempeh", "soy milk",
#     ],

#     # ── SEAFOOD (WESTERN) ────────────────────────────────────
#     "seafood_western": [
#         "grilled salmon", "baked cod", "tuna steak",
#         "lobster", "shrimp cocktail", "grilled shrimp",
#         "fish tacos", "crab legs", "oysters",
#         "clams", "mussels", "scallops",
#         "fish fillet", "tuna salad", "sardines",
#         "smoked salmon", "calamari", "shrimp scampi",
#     ],

#     # ── DELI & COLD CUTS ─────────────────────────────────────
#     "deli_cold_cuts": [
#         "deli turkey", "ham slice", "salami", "pepperoni",
#         "bologna", "pastrami", "roast beef slice",
#         "prosciutto", "mortadella", "deli meat platter",
#         "luncheon meat", "sausage", "bratwurst",
#         "chorizo", "kielbasa",
#     ],

#     # ── DESSERTS (BAKED) ─────────────────────────────────────
#     "desserts_baked": [
#         "cake", "chocolate cake", "vanilla cake", "red velvet cake",
#         "black forest cake", "carrot cake", "cheesecake",
#         "brownie", "cupcake", "muffin", "apple pie",
#         "blueberry pie", "chocolate mousse", "pastry", "eclair",
#         "tiramisu", "panna cotta", "creme brulee", "tart",
#         "profiterole", "cannoli", "macaron", "churros",
#         "doughnut", "cinnamon roll", "baklava", "cookie",
#         "chocolate chip cookie", "shortbread",
#     ],

#     # ── DESSERTS (FROZEN) ────────────────────────────────────
#     "desserts_frozen": [
#         "ice cream", "ice cream sundae", "gelato", "frozen yogurt",
#         "sorbet", "popsicle", "ice cream sandwich",
#         "ice cream cone", "milkshake ice cream",
#     ],

#     # ── BEVERAGES (COLD) ─────────────────────────────────────
#     "beverages_cold": [
#         "milkshake", "smoothie", "juice", "orange juice",
#         "mango juice", "lemonade", "iced tea", "cold coffee",
#         "protein shake", "bubble tea", "sparkling water",
#         "coconut water", "energy drink", "sports drink",
#         "iced latte", "frappuccino", "horchata",
#         "agua fresca", "kombucha",
#     ],

#     # ── BEVERAGES (HOT) ──────────────────────────────────────
#     "beverages_hot": [
#         "coffee", "black coffee", "espresso", "cappuccino",
#         "latte", "americano", "mocha", "hot chocolate",
#         "green tea", "black tea", "herbal tea",
#         "matcha latte", "turmeric latte", "golden milk",
#     ],

#     # ── ALCOHOLIC BEVERAGES ──────────────────────────────────
#     "beverages_alcoholic": [
#         "beer", "wine", "red wine", "white wine",
#         "cocktail", "margarita", "mojito", "whiskey glass",
#         "vodka", "gin and tonic", "champagne",
#         "sangria", "mimosa", "bloody mary",
#     ],

#     # ── FRUITS ───────────────────────────────────────────────
#     "fruits": [
#         "apple", "banana", "orange", "mango", "grapes",
#         "pineapple", "watermelon", "papaya", "strawberry",
#         "blueberry", "kiwi", "peach", "pomegranate",
#         "pear", "plum", "cherry", "raspberry", "lychee",
#         "dragon fruit", "guava", "coconut", "fig",
#         "apricot", "dates", "avocado", "lemon", "lime",
#         "grapefruit", "tangerine", "melon",
#     ],

#     # ── VEGETABLES ───────────────────────────────────────────
#     "vegetables": [
#         "tomato", "onion", "potato", "carrot", "cucumber",
#         "capsicum", "bell pepper", "broccoli", "cauliflower",
#         "cabbage", "spinach", "lettuce", "peas", "corn",
#         "eggplant", "zucchini", "mushroom", "garlic", "ginger",
#         "celery", "asparagus", "artichoke", "beetroot",
#         "radish", "turnip", "sweet potato", "yam",
#         "okra", "leek", "green beans", "snow peas",
#         "bok choy", "kale", "arugula", "fennel",
#         "jalapeño", "serrano pepper", "spring onion",
#     ],
# }

# ALL_LABELS = [label for labels in CATEGORIES.values() for label in labels]

# CATEGORY_DISPLAY = {
#     "indian_bread":                      "Indian flatbread like roti, naan, puri or bhatura",
#     "indian_curry_paneer":               "Indian paneer curry dish",
#     "indian_curry_dal":                  "Indian dal, chole or lentil dish in a bowl",
#     "indian_curry_veg":                  "Indian vegetable curry",
#     "indian_curry_chicken":              "Indian chicken curry",
#     "indian_curry_meat":                 "Indian mutton or meat curry",
#     "indian_curry_seafood":              "Indian seafood or egg curry",
#     "tandoor_grill":                     "tandoor grilled Indian food",
#     "rice_dishes":                       "rice dish like biryani or pulao",
#     "south_indian":                      "South Indian food like dosa or idli",
#     "indian_snacks_street":              "Indian street food like samosa, pani puri, chole bhature or pav bhaji",
#     "indian_sweets":                     "Indian sweet or dessert",
#     "indian_beverages":                  "Indian beverage like chai or lassi",
#     "western_pizza_pasta":               "pizza or pasta dish",
#     "western_burgers_sandwiches":        "burger, sandwich, hot dog, wrap or tacos",
#     "western_chicken":                   "Western fried or grilled chicken dish",
#     "western_mains":                     "Western main course like steak, roast or pork chops",
#     "western_sides":                     "Western side dish like fries, potato or garlic bread",
#     "soups_stews":                       "soup or stew in a bowl",
#     "asian_noodles":                     "Asian noodles, ramen, dumplings or spring rolls",
#     "asian_mains":                       "Asian main dish like sushi, teriyaki or curry",
#     "mexican_latin":                     "Mexican or Latin food like tacos, burritos or empanadas",
#     "middle_eastern_mediterranean":      "Middle Eastern or Mediterranean food like hummus, falafel or shawarma",
#     "breakfast":                         "breakfast food like eggs, pancakes or oatmeal",
#     "dairy_eggs":                        "dairy product or egg dish like yogurt, cheese or deviled eggs",
#     "breads_grains_western":             "Western bread or grain like sourdough, baguette or tortilla",
#     "salads":                            "fresh salad",
#     "snacks_processed":                  "packaged snack like chips, popcorn or crackers",
#     "nuts_seeds":                        "nuts, seeds or nut butter",
#     "legumes_beans":                     "legume or bean dish like chickpeas, lentils or tofu",
#     "seafood_western":                   "Western seafood like salmon, shrimp or lobster",
#     "deli_cold_cuts":                    "deli meat or cold cut like ham, salami or sausage",
#     "desserts_baked":                    "baked dessert like cake, brownie or pie",
#     "desserts_frozen":                   "frozen dessert like ice cream or gelato",
#     "beverages_cold":                    "cold beverage, smoothie or juice",
#     "beverages_hot":                     "hot beverage like coffee, tea or hot chocolate",
#     "beverages_alcoholic":               "alcoholic beverage like beer, wine or cocktail",
#     "fruits":                            "fresh fruit",
#     "vegetables":                        "raw or fresh vegetable",
# }

# # ─────────────────────────────────────────────────────────────
# # FOOD / NON-FOOD GATE TEXTS
# # ─────────────────────────────────────────────────────────────

# FOOD_TEXTS = [
#     "a photo of food",
#     "a plate of food",
#     "a delicious meal",
#     "indian food on a plate",
#     "street food being served",
# ]

# NON_FOOD_TEXTS = [
#     "a photo of a person",
#     "a photo of a car",
#     "a photo of a building",
#     "a random object",
#     "a blank background",
# ]

# # ─────────────────────────────────────────────────────────────
# # MODEL CACHE
# # ─────────────────────────────────────────────────────────────

# _finetuned_clip = None
# _general_clip   = None
# _clip_processor = None
# _tokenizer      = None
# _text_model     = None
# _yolo_model     = None

# # ─────────────────────────────────────────────────────────────
# # LOADERS
# # ─────────────────────────────────────────────────────────────

# def _get_clip_processor():
#     global _clip_processor
#     if _clip_processor is None:
#         print("[Loading CLIP processor]")
#         _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return _clip_processor


# def _get_general_clip():
#     global _general_clip
#     if _general_clip is None:
#         print("[Loading general CLIP]")
#         _general_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         _general_clip.eval()
#     return _general_clip


# def _get_finetuned_clip():
#     global _finetuned_clip
#     if _finetuned_clip is None:
#         if os.path.exists(FINETUNED_CLIP_PATH):
#             print("[Loading fine-tuned CLIP]")
#             checkpoint = torch.load(FINETUNED_CLIP_PATH, map_location="cpu")
#             state_dict = checkpoint["model"]
#             model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#             model.load_state_dict(state_dict, strict=False)
#             model.eval()
#             _finetuned_clip = model
#         else:
#             print("[WARNING] Fine-tuned CLIP checkpoint not found, using general CLIP only")
#             _finetuned_clip = "missing"
#     return None if _finetuned_clip == "missing" else _finetuned_clip


# def _get_yolo():
#     global _yolo_model
#     if _yolo_model is None:
#         print("[Loading YOLOv8]")
#         _yolo_model = YOLO("yolov8n.pt")
#     return _yolo_model


# def _get_text_model():
#     global _tokenizer, _text_model
#     if _text_model is None:
#         print("[Loading FLAN-T5]")
#         model_name  = "google/flan-t5-base"
#         _tokenizer  = AutoTokenizer.from_pretrained(model_name)
#         _text_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return _tokenizer, _text_model

# # ─────────────────────────────────────────────────────────────
# # PROMPT BUILDER — context-aware, 5 prompts per label
# # ─────────────────────────────────────────────────────────────

# _RAW_LABELS = set(
#     CATEGORIES["fruits"]
#     + CATEGORIES["vegetables"]
#     + CATEGORIES["nuts_seeds"]
# )
# PROMPTS_PER_LABEL = 5   # must match _make_prompts output length

# def _make_prompts(label: str) -> list:
#     """
#     5 prompts per label.
#     Raw produce / nuts get fresh prompts; cooked food gets serving prompts.
#     """
#     if label in _RAW_LABELS:
#         return [
#             f"a photo of {label}",
#             f"a fresh {label}",
#             f"raw {label}",
#             f"a close-up of {label}",
#             f"{label} on a white background",
#         ]
#     else:
#         return [
#             f"a photo of {label}",
#             f"a plate of {label}",
#             f"a close-up of {label}",
#             f"a delicious serving of {label}",
#             f"freshly made {label}",
#         ]

# # ─────────────────────────────────────────────────────────────
# # CORE: COSINE-SIMILARITY CLIP CLASSIFY
# # ─────────────────────────────────────────────────────────────

# def _clip_classify(model, image, labels: list, top_k: int = 3) -> tuple:
#     """
#     Classify image against labels using cosine similarity (NOT softmax).
#     Aggregates over PROMPTS_PER_LABEL prompts per label by taking the max.

#     Returns:
#         top_labels  — list of str, length top_k
#         top_scores  — list of float, cosine similarity in [~0.10, ~0.40]
#     """
#     processor   = _get_clip_processor()
#     all_prompts = []
#     for lbl in labels:
#         all_prompts.extend(_make_prompts(lbl))

#     inputs = processor(
#         text=all_prompts,
#         images=image,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#     )

#     with torch.no_grad():
#         image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
#         text_features  = model.get_text_features(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#         )

#     # L2-normalize both
#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

#     # Cosine similarities: shape (num_prompts,)
#     similarities = (image_features @ text_features.T)[0]

#     # Aggregate per label: max over its PROMPTS_PER_LABEL prompts
#     scores = []
#     for i, lbl in enumerate(labels):
#         start = i * PROMPTS_PER_LABEL
#         end   = start + PROMPTS_PER_LABEL
#         score = similarities[start:end].max().item()
#         scores.append((lbl, score))

#     scores.sort(key=lambda x: x[1], reverse=True)
#     top = scores[:top_k]
#     return [l for l, _ in top], [s for _, s in top]

# # ─────────────────────────────────────────────────────────────
# # TWO-STAGE CLASSIFICATION: coarse → fine
# # ─────────────────────────────────────────────────────────────

# def _clip_classify_two_stage(model, image, top_k: int = 3) -> tuple:
#     """
#     Stage 1 — pick top 3 coarse categories using short descriptive prompts.
#     Stage 2 — fine-grained cosine-sim classify only within those candidates.
#     Falls back to ALL_LABELS if no category clears the minimum threshold.
#     """
#     processor  = _get_clip_processor()
#     cat_names  = list(CATEGORIES.keys())
#     cat_texts  = [f"a photo of {CATEGORY_DISPLAY[c]}" for c in cat_names]

#     inputs = processor(
#         text=cat_texts,
#         images=image,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#     )

#     with torch.no_grad():
#         img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
#         txt_feat = model.get_text_features(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#         )

#     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
#     txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
#     cat_sims = (img_feat @ txt_feat.T)[0]

#     top2_idx    = cat_sims.topk(3).indices.tolist()
#     top2_scores = [cat_sims[i].item() for i in top2_idx]

#     COARSE_MIN = 0.15
#     if top2_scores[0] < COARSE_MIN:
#         print(f"[Two-stage] Coarse sim too low ({top2_scores[0]:.3f}), "
#               f"falling back to ALL_LABELS")
#         return _clip_classify(model, image, ALL_LABELS, top_k=top_k)

#     candidate_labels = []
#     for idx in top2_idx:
#         candidate_labels.extend(CATEGORIES[cat_names[idx]])

#     print(f"[Two-stage] Top categories : {[cat_names[i] for i in top2_idx]}")
#     print(f"[Two-stage] Cat sims       : {[round(s, 3) for s in top2_scores]}")
#     print(f"[Two-stage] Candidates     : {candidate_labels}")

#     return _clip_classify(model, image, candidate_labels, top_k=top_k)

# # ─────────────────────────────────────────────────────────────
# # FOOD / NON-FOOD GATE
# # ─────────────────────────────────────────────────────────────

# def _is_food(image) -> tuple:
#     """
#     Returns (is_food: bool, food_confidence: float).
#     Uses mean cosine similarity so score is not diluted by label count.
#     """
#     model     = _get_general_clip()
#     processor = _get_clip_processor()

#     all_texts = FOOD_TEXTS + NON_FOOD_TEXTS
#     inputs    = processor(
#         text=all_texts, images=image,
#         return_tensors="pt", padding=True,
#     )

#     with torch.no_grad():
#         img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
#         txt_feat = model.get_text_features(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#         )

#     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
#     txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
#     sims     = (img_feat @ txt_feat.T)[0]

#     food_conf     = sims[:len(FOOD_TEXTS)].mean().item()
#     non_food_conf = sims[len(FOOD_TEXTS):].mean().item()

#     return food_conf > non_food_conf, round(food_conf, 3)

# # ─────────────────────────────────────────────────────────────
# # IMAGE HELPERS
# # ─────────────────────────────────────────────────────────────

# def _crop_image(image, box, pad: int = 10):
#     x1, y1, x2, y2 = map(int, box)
#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(image.width,  x2 + pad)
#     y2 = min(image.height, y2 + pad)
#     return image.crop((x1, y1, x2, y2))

# # ─────────────────────────────────────────────────────────────
# # CAPTION GENERATOR
# # ─────────────────────────────────────────────────────────────

# def generate_caption(label: str) -> str:
#     tokenizer, model = _get_text_model()
#     prompt = (
#         f"Describe the food item '{label}' in a simple, structured way like a food dataset. "
#         f"Include: main ingredients, cooking method, texture or form. "
#         f"Keep it short, factual, and one sentence only."
#     )
#     inputs  = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=30)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ─────────────────────────────────────────────────────────────
# # DEBUG HELPER
# # ─────────────────────────────────────────────────────────────

# def _debug_detection(idx, box, yolo_conf, label, clip_conf):
#     print("\n" + "=" * 50)
#     print(f"[OBJECT {idx}]")
#     print(f"  Box           : {[round(v, 1) for v in box]}")
#     print(f"  YOLO conf     : {round(yolo_conf, 3)}")
#     print(f"  Predicted food: {label}")
#     print(f"  CLIP cos-sim  : {round(clip_conf, 3)}")
#     print("=" * 50)

# # ─────────────────────────────────────────────────────────────
# # CASE 1 HELPER — smart deduplication for single-plate images
# # ─────────────────────────────────────────────────────────────

# def _is_duplicate(new_label: str, accepted_labels: set) -> bool:
#     """
#     Returns True if new_label shares ANY word with an already-accepted label.
#     """
#     new_words = set(new_label.lower().split())
#     for seen_lbl in accepted_labels:
#         seen_words = set(seen_lbl.lower().split())
#         if new_words & seen_words:
#             print(f"  [Dedup] Skipping '{new_label}' (overlaps with '{seen_lbl}')")
#             return True
#     return False


# def _build_foods_case1(labels: list, scores: list, image_w: int, image_h: int) -> list:
#     """
#     Filters and deduplicates CLIP results for single-plate (Case 1) images.

#     Rules applied in order:
#       1. Skip if score < MIN_SINGLE_CONF
#       2. Stop if gap between rank-1 and current > TOP_GAP_THRESHOLD
#       3. Skip if label shares a word with any already-accepted label
#       4. Hard cap at 3 items
#     """
#     foods = []
#     seen  = set()

#     for idx, (lbl, score) in enumerate(zip(labels, scores)):

#         # Rule 1: minimum confidence
#         if score < MIN_SINGLE_CONF:
#             print(f"  [Conf filter] '{lbl}' score {score:.3f} < {MIN_SINGLE_CONF}, stopping")
#             break

#         # Rule 2: gap filter — top item has a clear lead
#         if idx > 0 and (scores[0] - score) > TOP_GAP_THRESHOLD:
#             print(f"  [Gap filter] Stopping at '{lbl}': "
#                   f"gap={scores[0] - score:.3f} > {TOP_GAP_THRESHOLD}")
#             break

#         # Rule 3: word-overlap deduplication
#         if _is_duplicate(lbl, seen):
#             continue

#         seen.add(lbl)
#         foods.append({
#             "label":      lbl,
#             "confidence": round(score, 3),
#             "caption":    f"a delicious plate of {lbl}",
#             "box":        [0, 0, image_w, image_h],
#         })

#         # Rule 4: hard cap
#         if len(foods) >= 3:
#             print("  [Hard cap] Reached 3 items, stopping")
#             break

#     return foods

# # ─────────────────────────────────────────────────────────────
# # MAIN PIPELINE
# # ─────────────────────────────────────────────────────────────

# def process_image(image: Image.Image) -> dict:
#     """
#     Full pipeline:
#       1. Food / non-food gate
#       2. YOLO object detection
#       3a. YOLO unreliable (<=1 box) → two-stage CLIP on full image
#       3b. YOLO reliable  (>1 boxes) → crop each box, two-stage CLIP per crop

#     Returns:
#       {
#         status  : "food" | "not_food",
#         primary : str,
#         count   : int,
#         foods   : [ {label, confidence, caption, box}, ... ]
#       }
#     """
#     image = image.convert("RGB")

#     # ── Step 1: Food gate ────────────────────────────────────
#     is_food_flag, food_conf = _is_food(image)
#     print(f"\n[Food gate] is_food={is_food_flag}  food_conf={food_conf}")

#     if not is_food_flag:
#         return {
#             "status":  "not_food",
#             "message": "This is not a food image",
#         }

#     # ── Step 2: YOLO detection ───────────────────────────────
#     yolo        = _get_yolo()
#     results     = yolo(image, conf=YOLO_CONF)
#     boxes       = results[0].boxes
#     valid_boxes = []

#     if boxes is not None:
#         for box_obj in boxes:
#             conf = float(box_obj.conf[0])
#             if conf > YOLO_VALID_CONF:
#                 valid_boxes.append(box_obj)

#     num_objects = len(valid_boxes)
#     print(f"\n{'#'*60}")
#     print(f"[YOLO] Valid boxes (conf > {YOLO_VALID_CONF}): {num_objects}")
#     print(f"{'#'*60}\n")

#     # ── Case 1: YOLO unreliable → full-image two-stage CLIP ──
#     if num_objects <= 1:
#         print("🚨 YOLO unreliable → two-stage CLIP on full image")

#         model          = _get_finetuned_clip() or _get_general_clip()
#         labels, scores = _clip_classify_two_stage(model, image, top_k=5)

#         print(f"\n[Case 1] Raw CLIP results:")
#         for lbl, sc in zip(labels, scores):
#             print(f"  {lbl:30s}  {sc:.3f}")

#         foods = _build_foods_case1(labels, scores, image.width, image.height)
#         foods.sort(key=lambda x: x["confidence"], reverse=True)

#         print(f"\n[Case 1] Final detections: {[f['label'] for f in foods]}")

#         return {
#             "status":  "food",
#             "primary": foods[0]["label"] if foods else "unknown",
#             "count":   len(foods),
#             "foods":   foods,
#         }

#     # ── Case 2: YOLO reliable → crop + two-stage CLIP ────────
#     print("✅ YOLO reliable → multi-object pipeline")

#     results_list = []
#     finetuned    = _get_finetuned_clip()

#     for i, box_obj in enumerate(valid_boxes):
#         box       = box_obj.xyxy[0].tolist()
#         yolo_conf = float(box_obj.conf[0])
#         crop      = _crop_image(image, box)

#         # Try fine-tuned CLIP first
#         if finetuned is not None:
#             labels, scores = _clip_classify_two_stage(finetuned, crop, top_k=3)
#             label, clip_conf = labels[0], scores[0]
#             _debug_detection(i + 1, box, yolo_conf, label, clip_conf)

#             if clip_conf >= TIER1_CONF:
#                 results_list.append({
#                     "food":       label,
#                     "confidence": round(clip_conf, 3),
#                     "box":        box,
#                 })
#                 continue

#         # Fallback: general CLIP
#         labels, scores = _clip_classify_two_stage(_get_general_clip(), crop, top_k=3)
#         label, clip_conf = labels[0], scores[0]
#         _debug_detection(i + 1, box, yolo_conf, label, clip_conf)

#         if clip_conf >= TIER2_CONF:
#             results_list.append({
#                 "food":       label,
#                 "confidence": round(clip_conf, 3),
#                 "box":        box,
#             })
#         else:
#             print(f"  ⚠️  Skipping object {i+1}: "
#                   f"clip_conf {clip_conf:.3f} < TIER2_CONF {TIER2_CONF}")

#     return {
#         "status":  "food",
#         "primary": results_list[0]["food"] if results_list else "unknown",
#         "count":   len(results_list),
#         "foods": [
#             {
#                 "label":      item["food"],
#                 "confidence": item["confidence"],
#                 "caption":    f"a delicious plate of {item['food']}",
#                 "box":        item["box"],
#             }
#             for item in results_list
#         ],
#     }


# import os
# import torch
# from PIL import Image
# from transformers import (
#     CLIPProcessor,
#     CLIPModel,
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM
# )
# from ultralytics import YOLO
# import numpy as np
# # ─────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────

# FINETUNED_CLIP_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "checkpoint_best.pt"
# )

# TIER1_CONF = 0.30
# TIER2_CONF = 0.22

# device = "cpu"
# torch.set_num_threads(4)

# # ─────────────────────────────────────────────────────────────
# # LABELS
# # ─────────────────────────────────────────────────────────────

# INDIAN_LABELS = [
#     "a photo of indian roti flatbread","a bowl of paneer curry with rich gravy","a plate of steamed basmati rice","a serving of indian dal lentil curry",
#     "a crispy dosa on a plate","a deep fried vada on a plate","a bowl of sambar lentil soup","a bowl of coconut chutney","idli served with chutney and sambar","a plate of indian vegetable sabzi",
#     "soft steamed idli","a plain dosa","a masala dosa filled with potato stuffing","a fried vada snack","a medu vada savory donut","a thick uttapam with toppings",
#     "a plate of biryani rice dish","a plate of chicken biryani","a plate of fried rice","a plate of paneer fried rice",
#     "a plate of vegetable pulao","chole bhature with chickpea curry and poori","pav bhaji mashed vegetable curry served with bread rolls and butter","a crispy samosa snack","a fried aloo tikki patty",
#     "paneer butter masala with visible paneer cubes in creamy tomato gravy","palak paneer with spinach gravy","dal makhani with creamy lentils",
#     "rajma kidney bean curry","aloo gobi with potato and cauliflower","butter chicken in rich tomato gravy","tandoori chicken grilled with spices",
#     "indian roti bread","soft naan bread","stuffed paratha flatbread","deep fried puri bread","fluffy bhatura bread",
#     "gulab jamun soaked in sugar syrup","rasgulla in sugar syrup","crispy jalebi dessert","sweet laddu balls","kheer rice pudding",
#     "sweet halwa dessert","poha flattened rice dish","upma semolina breakfast dish","dhokla steamed savory cake","pani puri street food snack",
#     "a bowl of sambar","a bowl of chutney","a bowl of indian curry","a golden brown puffed deep fried poori bread with crispy texture","a golden brown puffed deep fried poori bread with crispy texture"
# ]
# GLOBAL_LABELS = [
#     "a slice of pizza with toppings","a burger with bun and patty","a plate of pasta with sauce","a sandwich with fillings","a wrap with vegetables or meat",
#     "a bowl of noodles","a plate of fried noodles","sushi rolls on a plate","a bowl of fresh salad","fried chicken pieces","grilled chicken with seasoning",
#     "chicken wings with sauce","a serving of french fries","an omelette made with eggs","scrambled eggs on a plate","a stack of pancakes",
#     "waffles with syrup","tacos with filling","a hot dog in a bun","spring rolls with filling","dumplings on a plate","a steak piece of meat","a bowl of soup","a bowl of ramen noodles",
#     "a scoop of ice cream","a slice of cake","a chocolate cake slice","cookies on a plate","a donut with icing"
# ]
# ALL_LABELS = INDIAN_LABELS + GLOBAL_LABELS

# FOOD_TEXTS = [
#     "a photo of food", "a plate of food",
#     "a delicious meal", "indian food", "street food",
# ]

# NON_FOOD_TEXTS = [
#     "a person", "a car", "a building",
#     "random object", "blank background",
# ]

# # ─────────────────────────────────────────────────────────────
# # MODEL CACHE
# # ─────────────────────────────────────────────────────────────

# _finetuned_clip = None
# _general_clip = None
# _clip_processor = None

# _tokenizer = None
# _text_model = None
# yolo_conf=0.15
# # ─────────────────────────────────────────────────────────────
# # LOADERS
# # ─────────────────────────────────────────────────────────────
# _yolo_model = None

# def _get_yolo():
#     global _yolo_model
#     if _yolo_model is None:
#         print("[Loading YOLOv8]")
#         _yolo_model = YOLO("yolov8n.pt")  # or your trained model
#     return _yolo_model

# def _crop_image(image, box, pad=10):
#     x1, y1, x2, y2 = map(int, box)

#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(image.width, x2 + pad)
#     y2 = min(image.height, y2 + pad)

#     return image.crop((x1, y1, x2, y2))

# def _get_clip_processor():
#     global _clip_processor
#     if _clip_processor is None:
#         _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return _clip_processor


# def _get_finetuned_clip():
#     global _finetuned_clip

#     if _finetuned_clip is None:
#         if os.path.exists(FINETUNED_CLIP_PATH):
#             print("[Loading fine-tuned CLIP properly]")

#             checkpoint = torch.load(FINETUNED_CLIP_PATH, map_location="cpu")

#             # 🔥 KEY FIX
#             state_dict = checkpoint["model"]   # <-- extract actual model

#             model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#             model.load_state_dict(state_dict, strict=False)  # allow mismatch
#             model.eval()

#             _finetuned_clip = model
#         else:
#             _finetuned_clip = "missing"

#     return None if _finetuned_clip == "missing" else _finetuned_clip
# def _get_general_clip():
#     global _general_clip
#     if _general_clip is None:
#         print("[Loading general CLIP]")
#         _general_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         _general_clip.eval()
#     return _general_clip


# def _get_text_model():
#     global _tokenizer, _text_model
#     if _text_model is None:
#         print("[Loading FLAN-T5]")
#         model_name = "google/flan-t5-base"

#         _tokenizer = AutoTokenizer.from_pretrained(model_name)
#         _text_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     return _tokenizer, _text_model

# # ─────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────

# def _make_prompts(labels):
#     prompts = []
#     for lbl in labels:
#         prompts.extend([
#             f"a photo of {lbl}",
#             f"a plate of {lbl}",
#             f"a delicious {lbl}",
#             f"indian food {lbl}",
#         ])
#     return prompts

# def _clip_classify(model, image, labels, top_k=3):
#     processor = _get_clip_processor()
#     prompts = _make_prompts(labels)

#     inputs = processor(
#         text=prompts,
#         images=image,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = outputs.logits_per_image.softmax(dim=1)[0]

#     scores = []
#     for i, lbl in enumerate(labels):
#         score = probs[i*4:(i+1)*4].max().item()
#         scores.append((lbl, score))

#     scores.sort(key=lambda x: x[1], reverse=True)

#     return [l for l, _ in scores[:top_k]], [s for _, s in scores[:top_k]]
# def debug_print_detection(idx, box, yolo_conf, label, clip_conf):
#     print("\n" + "="*50)
#     print(f"[OBJECT {idx}]")
#     print(f"📦 Box: {box}")
#     print(f"🎯 YOLO Confidence: {round(yolo_conf, 3)}")
#     print(f"🍽️ Predicted Food: {label}")
#     print(f"🧠 CLIP Confidence: {round(clip_conf, 3)}")
#     print("="*50)

# def _is_food(image):
#     model = _get_general_clip()
#     processor = _get_clip_processor()

#     texts = FOOD_TEXTS + NON_FOOD_TEXTS
#     inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = outputs.logits_per_image.softmax(dim=1)[0]

#     food_conf = probs[:len(FOOD_TEXTS)].sum().item()
#     non_food_conf = probs[len(FOOD_TEXTS):].sum().item()

#     return food_conf > non_food_conf, round(food_conf, 3)


# def generate_caption(label):
#     tokenizer, model = _get_text_model()

#     # prompt = f"What is {label}? Describe it like a food menu in one sentence."
#     prompt = f"""
#     Describe the food item '{label}' in a simple, structured way like a food dataset.

#     Include:
#     - main ingredients
#     - cooking method
#     - texture or form

# Keep it short, factual, and one sentence only.
# """

#     inputs = tokenizer(prompt, return_tensors="pt")

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=30
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ─────────────────────────────────────────────────────────────
# # MAIN PIPELINE
# # ─────────────────────────────────────────────────────────────

# def process_image(image):
#     image = image.convert("RGB")

#     # ─────────────────────────────
#     # Step 1: Food check
#     # ─────────────────────────────
#     is_food_flag, food_conf = _is_food(image)
#     if not is_food_flag:
#         return {
#             "status": "not_food",
#             "message": "This is not a food image"
#         }

#     # ─────────────────────────────
#     # Step 2: YOLO detection
#     # ─────────────────────────────
#     yolo = _get_yolo()
#     results = yolo(image, conf=0.15)

#     boxes = results[0].boxes
#     valid_boxes = []

#     if boxes is not None:
#         for box_obj in boxes:
#             conf = float(box_obj.conf[0])
#             if conf > 0.4:   # strong detections only
#                 valid_boxes.append(box_obj)

#     num_objects = len(valid_boxes)

#     print("\n" + "#"*60)
#     print(f"[YOLO] Valid objects after filtering: {num_objects}")
#     print("#"*60)

#     # ─────────────────────────────
#     # CASE 1: YOLO unreliable → CLIP multi detection
#     # ─────────────────────────────
#     if num_objects <= 1:

#         print("\n🚨 YOLO unreliable → using CLIP multi-food detection")

#         labels, scores = _clip_classify(_get_general_clip(), image, ALL_LABELS, top_k=3)

#         foods = []
#         for l, s in zip(labels, scores):
#             if s > 0.12:
#                 foods.append({
#                     "label": l,
#                     "confidence": round(s, 3),
#                     "caption": f"a delicious plate of {l}",
#                     "box": [0, 0, image.width, image.height]  # ⚠️ FULL IMAGE fallback
#                 })

#         # remove duplicates
#         unique_foods = []
#         seen = set()
#         for f in foods:
#             key = f["label"].split()[-1]
#             if key not in seen:
#                 unique_foods.append(f)
#                 seen.add(key)

#         foods = sorted(unique_foods, key=lambda x: x["confidence"], reverse=True)

#         return {
#             "status": "food",
#             "primary": foods[0]["label"] if foods else "unknown",
#             "count": len(foods),
#             "foods": foods
#         }

#     # ─────────────────────────────
#     # CASE 2: YOLO reliable → crop + CLIP
#     # ─────────────────────────────
#     else:
#         print("\n✅ YOLO reliable → using multi-object pipeline")

#         results_list = []

#         for i, box_obj in enumerate(valid_boxes):

#             box = box_obj.xyxy[0].tolist()  # ✅ KEEP THIS
#             yolo_conf = float(box_obj.conf[0])

#             crop = _crop_image(image, box)

#             # Try fine-tuned CLIP
#             finetuned = _get_finetuned_clip()
#             if finetuned is not None:
#                 labels, scores = _clip_classify(finetuned, crop, ALL_LABELS)
#                 label, clip_conf = labels[0], scores[0]

#                 debug_print_detection(i+1, box, yolo_conf, label, clip_conf)

#                 if clip_conf >= TIER1_CONF:
#                     results_list.append({
#                         "food": label,
#                         "confidence": round(clip_conf, 3),
#                         "box": box   # ✅ IMPORTANT
#                     })
#                     continue

#             # fallback to general CLIP
#             labels, scores = _clip_classify(_get_general_clip(), crop, ALL_LABELS)
#             label, clip_conf = labels[0], scores[0]

#             debug_print_detection(i+1, box, yolo_conf, label, clip_conf)

#             results_list.append({
#                 "food": label,
#                 "confidence": round(clip_conf, 3),
#                 "box": box   # ✅ IMPORTANT
#             })

#         return {
#             "status": "food",
#             "primary": results_list[0]["food"] if results_list else "unknown",
#             "count": len(results_list),
#             "foods": [
#                 {
#                     "label": item["food"],
#                     "confidence": item["confidence"],
#                     "caption": f"a delicious plate of {item['food']}",
#                     "box": item["box"]   # ✅ PASS TO BACKEND
#                 }
#                 for item in results_list
#             ]
#         }

# import os
# import torch
# from PIL import Image
# from transformers import (
#     CLIPProcessor,
#     CLIPModel,
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM
# )
# from ultralytics import YOLO
# import numpy as np
# # ─────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────

# FINETUNED_CLIP_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "checkpoint_best.pt"
# )

# TIER1_CONF = 0.30
# TIER2_CONF = 0.22

# device = "cpu"
# torch.set_num_threads(4)

# # ─────────────────────────────────────────────────────────────
# # LABELS
# # ─────────────────────────────────────────────────────────────

# INDIAN_LABELS = [
#     "a photo of indian roti flatbread","a bowl of paneer curry with rich gravy","a plate of steamed basmati rice","a serving of indian dal lentil curry",
#     "a crispy dosa on a plate","a deep fried vada on a plate","a bowl of sambar lentil soup","a bowl of coconut chutney","idli served with chutney and sambar","a plate of indian vegetable sabzi",
#     "soft steamed idli","a plain dosa","a masala dosa filled with potato stuffing","a fried vada snack","a medu vada savory donut","a thick uttapam with toppings",
#     "a plate of biryani rice dish","a plate of chicken biryani","a plate of fried rice","a plate of paneer fried rice",
#     "a plate of vegetable pulao","chole bhature with chickpea curry and poori","pav bhaji mashed vegetable curry served with bread rolls and butter","a crispy samosa snack","a fried aloo tikki patty",
#     "paneer butter masala with visible paneer cubes in creamy tomato gravy","palak paneer with spinach gravy","dal makhani with creamy lentils",
#     "rajma kidney bean curry","aloo gobi with potato and cauliflower","butter chicken in rich tomato gravy","tandoori chicken grilled with spices",
#     "indian roti bread","soft naan bread","stuffed paratha flatbread","deep fried puri bread","fluffy bhatura bread",
#     "gulab jamun soaked in sugar syrup","rasgulla in sugar syrup","crispy jalebi dessert","sweet laddu balls","kheer rice pudding",
#     "sweet halwa dessert","poha flattened rice dish","upma semolina breakfast dish","dhokla steamed savory cake","pani puri street food snack",
#     "a bowl of sambar","a bowl of chutney","a bowl of indian curry","a golden brown puffed deep fried poori bread with crispy texture","a golden brown puffed deep fried poori bread with crispy texture"
# ]
# GLOBAL_LABELS = [
#     "a slice of pizza with toppings","a burger with bun and patty","a plate of pasta with sauce","a sandwich with fillings","a wrap with vegetables or meat",
#     "a bowl of noodles","a plate of fried noodles","sushi rolls on a plate","a bowl of fresh salad","fried chicken pieces","grilled chicken with seasoning",
#     "chicken wings with sauce","a serving of french fries","an omelette made with eggs","scrambled eggs on a plate","a stack of pancakes",
#     "waffles with syrup","tacos with filling","a hot dog in a bun","spring rolls with filling","dumplings on a plate","a steak piece of meat","a bowl of soup","a bowl of ramen noodles",
#     "a scoop of ice cream","a slice of cake","a chocolate cake slice","cookies on a plate","a donut with icing"
# ]
# ALL_LABELS = INDIAN_LABELS + GLOBAL_LABELS

# FOOD_TEXTS = [
#     "a photo of food", "a plate of food",
#     "a delicious meal", "indian food", "street food",
# ]

# NON_FOOD_TEXTS = [
#     "a person", "a car", "a building",
#     "random object", "blank background",
# ]

# # ─────────────────────────────────────────────────────────────
# # MODEL CACHE
# # ─────────────────────────────────────────────────────────────

# _finetuned_clip = None
# _general_clip = None
# _clip_processor = None

# _tokenizer = None
# _text_model = None
# yolo_conf=0.15
# # ─────────────────────────────────────────────────────────────
# # LOADERS
# # ─────────────────────────────────────────────────────────────
# _yolo_model = None

# def _get_yolo():
#     global _yolo_model
#     if _yolo_model is None:
#         print("[Loading YOLOv8]")
#         _yolo_model = YOLO("yolov8n.pt")  # or your trained model
#     return _yolo_model

# def _crop_image(image, box, pad=10):
#     x1, y1, x2, y2 = map(int, box)

#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(image.width, x2 + pad)
#     y2 = min(image.height, y2 + pad)

#     return image.crop((x1, y1, x2, y2))

# def _get_clip_processor():
#     global _clip_processor
#     if _clip_processor is None:
#         _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return _clip_processor


# def _get_finetuned_clip():
#     global _finetuned_clip

#     if _finetuned_clip is None:
#         if os.path.exists(FINETUNED_CLIP_PATH):
#             print("[Loading fine-tuned CLIP properly]")

#             checkpoint = torch.load(FINETUNED_CLIP_PATH, map_location="cpu")

#             # 🔥 KEY FIX
#             state_dict = checkpoint["model"]   # <-- extract actual model

#             model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#             model.load_state_dict(state_dict, strict=False)  # allow mismatch
#             model.eval()

#             _finetuned_clip = model
#         else:
#             _finetuned_clip = "missing"

#     return None if _finetuned_clip == "missing" else _finetuned_clip
# def _get_general_clip():
#     global _general_clip
#     if _general_clip is None:
#         print("[Loading general CLIP]")
#         _general_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         _general_clip.eval()
#     return _general_clip


# def _get_text_model():
#     global _tokenizer, _text_model
#     if _text_model is None:
#         print("[Loading FLAN-T5]")
#         model_name = "google/flan-t5-base"

#         _tokenizer = AutoTokenizer.from_pretrained(model_name)
#         _text_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     return _tokenizer, _text_model

# # ─────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────

# def _make_prompts(labels):
#     prompts = []
#     for lbl in labels:
#         prompts.extend([
#             f"a photo of {lbl}",
#             f"a plate of {lbl}",
#             f"a delicious {lbl}",
#             f"indian food {lbl}",
#         ])
#     return prompts

# def _clip_classify(model, image, labels, top_k=3):
#     processor = _get_clip_processor()
#     prompts = _make_prompts(labels)

#     inputs = processor(
#         text=prompts,
#         images=image,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=77,
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = outputs.logits_per_image.softmax(dim=1)[0]

#     scores = []
#     for i, lbl in enumerate(labels):
#         score = probs[i*4:(i+1)*4].max().item()
#         scores.append((lbl, score))

#     scores.sort(key=lambda x: x[1], reverse=True)

#     return [l for l, _ in scores[:top_k]], [s for _, s in scores[:top_k]]
# def debug_print_detection(idx, box, yolo_conf, label, clip_conf):
#     print("\n" + "="*50)
#     print(f"[OBJECT {idx}]")
#     print(f"📦 Box: {box}")
#     print(f"🎯 YOLO Confidence: {round(yolo_conf, 3)}")
#     print(f"🍽️ Predicted Food: {label}")
#     print(f"🧠 CLIP Confidence: {round(clip_conf, 3)}")
#     print("="*50)

# def _is_food(image):
#     model = _get_general_clip()
#     processor = _get_clip_processor()

#     texts = FOOD_TEXTS + NON_FOOD_TEXTS
#     inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = outputs.logits_per_image.softmax(dim=1)[0]

#     food_conf = probs[:len(FOOD_TEXTS)].sum().item()
#     non_food_conf = probs[len(FOOD_TEXTS):].sum().item()

#     return food_conf > non_food_conf, round(food_conf, 3)


# def generate_caption(label):
#     tokenizer, model = _get_text_model()

#     # prompt = f"What is {label}? Describe it like a food menu in one sentence."
#     prompt = f"""
#     Describe the food item '{label}' in a simple, structured way like a food dataset.

#     Include:
#     - main ingredients
#     - cooking method
#     - texture or form

# Keep it short, factual, and one sentence only.
# """

#     inputs = tokenizer(prompt, return_tensors="pt")

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=30
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ─────────────────────────────────────────────────────────────
# # MAIN PIPELINE
# # ─────────────────────────────────────────────────────────────

# def process_image(image):
#     image = image.convert("RGB")

#     # ─────────────────────────────
#     # Step 1: Food check
#     # ─────────────────────────────
#     is_food_flag, food_conf = _is_food(image)
#     if not is_food_flag:
#         return {
#             "status": "not_food",
#             "message": "This is not a food image"
#         }

#     # ─────────────────────────────
#     # Step 2: YOLO detection
#     # ─────────────────────────────
#     yolo = _get_yolo()
#     results = yolo(image, conf=0.15)

#     boxes = results[0].boxes
#     valid_boxes = []

#     if boxes is not None:
#         for box_obj in boxes:
#             conf = float(box_obj.conf[0])
#             if conf > 0.4:   # strong detections only
#                 valid_boxes.append(box_obj)

#     num_objects = len(valid_boxes)

#     print("\n" + "#"*60)
#     print(f"[YOLO] Valid objects after filtering: {num_objects}")
#     print("#"*60)

#     # ─────────────────────────────
#     # CASE 1: YOLO unreliable → CLIP multi detection
#     # ─────────────────────────────
#     if num_objects <= 1:

#         print("\n🚨 YOLO unreliable → using CLIP multi-food detection")

#         labels, scores = _clip_classify(_get_general_clip(), image, ALL_LABELS, top_k=3)

#         foods = []
#         for l, s in zip(labels, scores):
#             if s > 0.12:
#                 foods.append({
#                     "label": l,
#                     "confidence": round(s, 3),
#                     "caption": f"a delicious plate of {l}",
#                     "box": [0, 0, image.width, image.height]  # ⚠️ FULL IMAGE fallback
#                 })

#         # remove duplicates
#         unique_foods = []
#         seen = set()
#         for f in foods:
#             key = f["label"].split()[-1]
#             if key not in seen:
#                 unique_foods.append(f)
#                 seen.add(key)

#         foods = sorted(unique_foods, key=lambda x: x["confidence"], reverse=True)

#         return {
#             "status": "food",
#             "primary": foods[0]["label"] if foods else "unknown",
#             "count": len(foods),
#             "foods": foods
#         }

#     # ─────────────────────────────
#     # CASE 2: YOLO reliable → crop + CLIP
#     # ─────────────────────────────
#     else:
#         print("\n✅ YOLO reliable → using multi-object pipeline")

#         results_list = []

#         for i, box_obj in enumerate(valid_boxes):

#             box = box_obj.xyxy[0].tolist()  # ✅ KEEP THIS
#             yolo_conf = float(box_obj.conf[0])

#             crop = _crop_image(image, box)

#             # Try fine-tuned CLIP
#             finetuned = _get_finetuned_clip()
#             if finetuned is not None:
#                 labels, scores = _clip_classify(finetuned, crop, ALL_LABELS)
#                 label, clip_conf = labels[0], scores[0]

#                 debug_print_detection(i+1, box, yolo_conf, label, clip_conf)

#                 if clip_conf >= TIER1_CONF:
#                     results_list.append({
#                         "food": label,
#                         "confidence": round(clip_conf, 3),
#                         "box": box   # ✅ IMPORTANT
#                     })
#                     continue

#             # fallback to general CLIP
#             labels, scores = _clip_classify(_get_general_clip(), crop, ALL_LABELS)
#             label, clip_conf = labels[0], scores[0]

#             debug_print_detection(i+1, box, yolo_conf, label, clip_conf)

#             results_list.append({
#                 "food": label,
#                 "confidence": round(clip_conf, 3),
#                 "box": box   # ✅ IMPORTANT
#             })

#         return {
#             "status": "food",
#             "primary": results_list[0]["food"] if results_list else "unknown",
#             "count": len(results_list),
#             "foods": [
#                 {
#                     "label": item["food"],
#                     "confidence": item["confidence"],
#                     "caption": f"a delicious plate of {item['food']}",
#                     "box": item["box"]   # ✅ PASS TO BACKEND
#                 }
#                 for item in results_list
#             ]
#         }


import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
TIER1_CONF = 0.40
CLIP_MULTI_THRESHOLD = 0.25
YOLO_CONF = 0.40

device = "cpu"
torch.set_num_threads(4)

# ─────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────
INDIAN_LABELS = [
    "biryani", "veg biryani", "chicken biryani", "egg biryani","chicken biryani with egg","paneer biryani","mushroom biryani","hyderabadi biryani","mutton biryani",
    "pulao", "fried rice", "veg fried rice","chicken fried rice","egg fried rice","dum biryani","jeera rice","curd rice","lemon rice","coconut rice","tamarind rice",
    "butter chicken", "chicken curry", "chicken masala", "chicken gravy","fish curry","chole masala",
    "dal tadka", "dal makhani", "rajma", "chole","mixed veg curry",
    "paneer butter masala", "palak paneer", "paneer curry",
    "roti", "naan","butter naan","garlic naan","tandoori roti","aloo paratha", "paratha", "poori", "bhatura","dhokla","idiyappam",
    "idli", "dosa", "masala dosa","rava dosa","onion dosa", "vada", "uttapam", "upma","pongal",
    "pani puri", "bhel puri", "vada pav", "pav bhaji","sambar","rasam","sev puri","dahi puri","mirchi bajji","onion pakora","pakora","chaat",
    "samosa", "kachori","tomato chtney","green chutney","coconut chutney",
    "gulab jamun", "rasgulla", "jalebi","dhoodhpeda", "laddu", "kheer","rasmalai","raita","kaju katli","halwa","payasam","chai","masala chai","filter coffee","lassi","mango lassi","butter milk","sugarcane juice"
]

GLOBAL_LABELS = [
    "pizza","cheese pizza","pepperoni pizza","chicken pizza", "burger","veg burger","cheese burger", "sandwich", "wrap","pizza with toppings","grilled cheese sandwich","greek salad","churros"
    "pasta","white sauce pasta","red sauce pasta","alfredo pasta","penne pasta","mac and cheese", "spaghetti", "lasagna",
    "noodles","hakka noodles","chowmein","udon noodles", "ramen", "sushi", "dumplings","momos","steamed momos","fried momos",
    "taco", "burrito", "nachos","salad","fruit salad","smoothie",
    "fried chicken", "grilled chicken", "chicken wings","chicken nuggets","hot dog",
    "french fries","loaded fries", "omelette", "scrambled eggs","fried eggs",
    "pancakes", "waffles","toast","milk and cereal",
    "cake", "chocolate cake","red velvet cake", "donut","donut with icing","croissant", "cookies", "ice cream","vannila cake","carrot cake","cheesecake","ice cream","vanilla ice cream","chocolate ice cream","brownies","chocolate","cupcake","chocolate cupcake","macaroon","cookie","chocolate chip cookie",
    "pad thai","strawberry shortcake","tiramisu","gnocci","apple pie","ravioli"
]
VEGETABLE_LABELS = [
    "potato", "tomato", "onion", "carrot", "cabbage", "cauliflower", "broccoli",
    "spinach", "lettuce", "cucumber", "capsicum", "bell pepper", "green chili",
    "eggplant", "brinjal", "okra", "lady finger", "peas", "green beans",
    "corn", "sweet corn", "pumpkin", "bottle gourd", "ridge gourd",
    "bitter gourd", "zucchini", "radish", "beetroot", "turnip",
    "spring onion", "garlic", "ginger", "mushroom"
]

FRUIT_LABELS = [
    "apple", "banana", "mango", "orange", "grapes", "pineapple",
    "watermelon", "papaya", "guava", "pomegranate",
    "strawberry", "blueberry", "raspberry",
    "pear", "peach", "plum", "cherry",
    "kiwi", "dragon fruit", "jackfruit", "lychee",
    "coconut", "fig", "apricot"
]

SIMPLE_FOOD_LABELS = [
    "milk", "glass of milk", "cup of milk",
    "curd", "yogurt", "plain yogurt",
    "butter", "cheese", "paneer",
    "bread", "white bread", "brown bread",
    "toast", "buttered toast","tomato sauce",
    "boiled egg", "fried egg", "omelette",
    "rice", "plain rice", "steamed rice",
    "salt", "sugar", "jaggery", "honey",
    "tea", "black tea", "green tea",
    "coffee", "black coffee",
    "juice", "fruit juice",
    "water", "glass of water"
]

ALL_LABELS = INDIAN_LABELS + GLOBAL_LABELS + VEGETABLE_LABELS + FRUIT_LABELS + SIMPLE_FOOD_LABELS

# ─────────────────────────────────────────
# FOOD CHECK PROMPTS
# ─────────────────────────────────────────
FOOD_TEXTS = [
    "a photo of food",
    "a delicious meal",
    "a plate of food",
]

NON_FOOD_TEXTS = [
    "a car","a building","a laptop","a person"
]

# ─────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────
_yolo = None
_clip = None
_processor = None

def get_yolo():
    global _yolo
    if _yolo is None:
        print("[Loading YOLO]")
        _yolo = YOLO("yolov8n.pt")
    return _yolo

def get_clip():
    global _clip
    if _clip is None:
        print("[Loading CLIP]")
        _clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip.eval()
    return _clip

def get_processor():
    global _processor
    if _processor is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _processor

# ─────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────
def make_prompts(labels):
    prompts = []
    for lbl in labels:
        prompts.extend([
            f"a photo of {lbl}",
            f"a plate of {lbl}",
            f"a bowl of {lbl}",
            f"a close-up of {lbl}",
        ])
    return prompts

# ─────────────────────────────────────────
# CLIP CLASSIFIER
# ─────────────────────────────────────────
def clip_classify(image, labels, top_k=5):
    model = get_clip()
    processor = get_processor()

    prompts = make_prompts(labels)

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]

    scores = []
    for i, lbl in enumerate(labels):
        score = probs[i*4:(i+1)*4].max().item()
        scores.append((lbl, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# ─────────────────────────────────────────
# FOOD CHECK
# ─────────────────────────────────────────
def is_food(image):
    model = get_clip()
    processor = get_processor()

    texts = FOOD_TEXTS + NON_FOOD_TEXTS

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]

    food_score = probs[:len(FOOD_TEXTS)].sum().item()
    non_food_score = probs[len(FOOD_TEXTS):].sum().item()

    return food_score > non_food_score

# ─────────────────────────────────────────
# CROP
# ─────────────────────────────────────────
def crop_image(image, box, pad=10):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)
    return image.crop((x1, y1, x2, y2))

# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def process_image(image):
    image = image.convert("RGB")

    # STEP 1: FOOD CHECK
    if not is_food(image):
        return {"status": "not_food"}

    # STEP 2: YOLO
    yolo = get_yolo()
    results = yolo(image, conf=YOLO_CONF)

    boxes = results[0].boxes
    valid_boxes = []

    if boxes is not None:
        for b in boxes:
            if float(b.conf[0]) >= YOLO_CONF:
                valid_boxes.append(b)

    num_objects = len(valid_boxes)

    # ─────────────────────────────
    # CASE 1: NO OBJECT → CLIP MULTI
    # ─────────────────────────────
    if num_objects == 0:
        preds = clip_classify(image, ALL_LABELS, top_k=5)

        foods = [
            {"label": l, "confidence": round(s, 3)}
            for l, s in preds if s >= CLIP_MULTI_THRESHOLD
        ]

        return {
            "status": "food",
            "mode": "multi",
            "primary": foods[0]["label"] if foods else "unknown",
            "count": len(foods),
            "foods": foods
        }

    # ─────────────────────────────
    # CASE 2: SINGLE OBJECT
    # ─────────────────────────────
    elif num_objects == 1:
        box = valid_boxes[0].xyxy[0].tolist()
        crop = crop_image(image, box)

        preds = clip_classify(crop, ALL_LABELS, top_k=3)

        # 🔥 IMPORTANT: detect if multiple foods exist
        multi_foods = [
            {"label": l, "confidence": round(s, 3)}
            for l, s in preds if s >= CLIP_MULTI_THRESHOLD
        ]

        if len(multi_foods) > 1:
            return {
                "status": "food",
                "mode": "multi",
                "primary": multi_foods[0]["label"],
                "count": len(multi_foods),
                "foods": multi_foods
            }

        # else single
        label, score = preds[0]

        return {
            "status": "food",
            "mode": "single",
            "primary": label,
            "confidence": round(score, 3),
            "count": 1,
            "foods": [{
                "label": label,
                "confidence": round(score, 3)
            }]
        }

    # ─────────────────────────────
    # CASE 3: MULTIPLE OBJECTS
    # ─────────────────────────────
    else:
        results_list = []

        for b in valid_boxes:
            box = b.xyxy[0].tolist()
            crop = crop_image(image, box)

            preds = clip_classify(crop, ALL_LABELS, top_k=1)
            label, score = preds[0]

            results_list.append({
                "label": label,
                "confidence": round(score, 3),
                "box": box
            })

        return {
            "status": "food",
            "mode": "multi",
            "primary": results_list[0]["label"],
            "count": len(results_list),
            "foods": results_list
        }