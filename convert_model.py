import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

# Load the configuration
config = AutoConfig.from_pretrained(".")
config.num_labels = 4
config.id2label = {0: "Legal", 1: "Finance", 2: "HR", 3: "Code"}
config.label2id = {"Legal": 0, "Finance": 1, "HR": 2, "Code": 3}

# Create a new model with the modified config
model = AutoModelForSequenceClassification.from_config(config)

# Load the pre-trained weights
state_dict = torch.load("pytorch_model.bin")

# Remove the classifier weights from the state dict as we'll use a new one
classifier_keys = [k for k in state_dict.keys() if "classifier" in k]
for k in classifier_keys:
    del state_dict[k]

# Load the modified state dict
model.load_state_dict(state_dict, strict=False)

# Save the modified model
torch.save(model.state_dict(), "pytorch_model.bin")
model.config.save_pretrained(".") 