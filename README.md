# Cato Topic Analyzer

## AI-Security-Squad

A powerful multilingual topic analysis tool that uses state-of-the-art language models to classify text into different categories of sensitive information.

### Features

- Multilingual support through mDeBERTa-v3 models
- Real-time text classification
- Interactive visualization of results
- Support for multiple classification categories:
  - Personal Financial Information
  - Company Financial Information
  - Human Resources and Employment
  - Legal Consulting
  - Health and Medical Information
  - Customer and Client Data
  - Code Consulting

### Setup

1. Clone the repository:
```bash
git clone https://github.com/cyber1337research/topic-analyzer.git
cd topic-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Models

The application supports two mDeBERTa-v3 models:
- MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

### Usage

1. Select a model from the dropdown menu
2. Choose input method:
   - Enter custom text
   - Select from examples
3. View the classification results in an interactive bar chart

### Requirements

- Python 3.8+
- Streamlit
- PyTorch
- Transformers
- Plotly

### License

MIT License

---
language:
- multilingual
- zh
- ja
- ar
- ko
- de
- fr
- es
- pt
- hi
- id
- it
- tr
- ru
- bn
- ur
- mr
- ta
- vi
- fa
- pl
- uk
- nl
- sv
- he
- sw
- ps
license: mit
tags:
- zero-shot-classification
- text-classification
- nli
- pytorch
datasets:
- MoritzLaurer/multilingual-NLI-26lang-2mil7
- xnli
- multi_nli
- facebook/anli
- fever
- lingnli
- alisawuffles/WANLI
metrics:
- accuracy
pipeline_tag: zero-shot-classification
widget:
- text: Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU
  candidate_labels: politics, economy, entertainment, environment
model-index:
- name: DeBERTa-v3-base-xnli-multilingual-nli-2mil7
  results:
  - task:
      type: text-classification
      name: Natural Language Inference
    dataset:
      name: MultiNLI-matched
      type: multi_nli
      split: validation_matched
    metrics:
    - type: accuracy
      value: 0,857
      verified: false
  - task:
      type: text-classification
      name: Natural Language Inference
    dataset:
      name: MultiNLI-mismatched
      type: multi_nli
      split: validation_mismatched
    metrics:
    - type: accuracy
      value: 0,856
      verified: false
  - task:
      type: text-classification
      name: Natural Language Inference
    dataset:
      name: ANLI-all
      type: anli
      split: test_r1+test_r2+test_r3
    metrics:
    - type: accuracy
      value: 0,537
      verified: false
  - task:
      type: text-classification
      name: Natural Language Inference
    dataset:
      name: ANLI-r3
      type: anli
      split: test_r3
    metrics:
    - type: accuracy
      value: 0,497
      verified: false
  - task:
      type: text-classification
      name: Natural Language Inference
    dataset:
      name: WANLI
      type: alisawuffles/WANLI
      split: test
    metrics:
    - type: accuracy
      value: 0,732
      verified: false
  - task:
      type: text-classification
      name: Natural Language Inference
    dataset:
      name: LingNLI
      type: lingnli
      split: test
    metrics:
    - type: accuracy
      value: 0,788
      verified: false
  - task:
      type: text-classification
      name: Natural Language Inference
    dataset:
      name: fever-nli
      type: fever-nli
      split: test
    metrics:
    - type: accuracy
      value: 0,761
      verified: false
---
# Model card for mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

## Model description

This multilingual model can perform natural language inference (NLI) on 100 languages and is therefore also suitable for multilingual zero-shot classification. The underlying mDeBERTa-v3-base model was pre-trained by Microsoft on the [CC100 multilingual dataset](https://huggingface.co/datasets/cc100) with 100 languages. The model was then fine-tuned on the [XNLI dataset](https://huggingface.co/datasets/xnli) and on the [multilingual-NLI-26lang-2mil7 dataset](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7). Both datasets contain more than 2.7 million hypothesis-premise pairs in 27 languages spoken by more than 4 billion people. 

As of December 2021, mDeBERTa-v3-base is the best performing multilingual base-sized transformer model introduced by Microsoft in [this paper](https://arxiv.org/pdf/2111.09543.pdf). 


### How to use the model
#### Simple zero-shot classification pipeline
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
sequence_to_classify = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)
```
#### NLI use-case
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
hypothesis = "Emmanuel Macron is the President of France"

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "neutral", "contradiction"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)
```

### Training data
This model was trained on the [multilingual-nli-26lang-2mil7 dataset](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7) and the [XNLI](https://huggingface.co/datasets/xnli) validation dataset. 

The multilingual-nli-26lang-2mil7 dataset contains 2 730 000 NLI hypothesis-premise pairs in 26 languages spoken by more than 4 billion people. The dataset contains 105 000 text pairs per language. It is based on the English datasets [MultiNLI](https://huggingface.co/datasets/multi_nli), [Fever-NLI](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md), [ANLI](https://huggingface.co/datasets/anli), [LingNLI](https://arxiv.org/pdf/2104.07179.pdf) and [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) and was created using the latest open-source machine translation models. The languages in the dataset are: ['ar', 'bn', 'de', 'es', 'fa', 'fr', 'he', 'hi', 'id', 'it', 'ja', 'ko', 'mr', 'nl', 'pl', 'ps', 'pt', 'ru', 'sv', 'sw', 'ta', 'tr', 'uk', 'ur', 'vi', 'zh'] (see [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes). For more details, see the [datasheet](XXX). In addition, a sample of 105 000 text pairs was also added for English following the same sampling method as the other languages, leading to 27 languages. 

Moreover, for each language a random set of 10% of the hypothesis-premise pairs was added where an English hypothesis was paired with the premise in the other language (and the same for English premises and other language hypotheses). This mix of languages in the text pairs should enable users to formulate a hypothesis in English for a target text in another language. 

The [XNLI](https://huggingface.co/datasets/xnli) validation set consists of 2490 professionally translated texts from English to 14 other languages (37350 texts in total) (see [this paper](https://arxiv.org/pdf/1809.05053.pdf)). Note that XNLI also contains a training set of 14 machine translated versions of the MultiNLI dataset for 14 languages, but this data was excluded due to quality issues with the machine translations from 2018. 

Note that for evaluation purposes, three languages were excluded from the XNLI training data and only included in the test data: ["bg","el","th"]. This was done in order to test the performance of the model on languages it has not seen during NLI fine-tuning on 27 languages, but only during pre-training on 100 languages - see evaluation metrics below. 

The total training dataset had a size of 3 287 280 hypothesis-premise pairs. 


### Training procedure
mDeBERTa-v3-base-mnli-xnli was trained using the Hugging Face trainer with the following hyperparameters.

```
training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    learning_rate=2e-05,
    per_device_train_batch_size=32,   # batch size per device during training
    gradient_accumulation_steps=2,   # to double the effective batch size for 
    warmup_ratio=0.06,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    fp16=False
)
```

### Eval results
The model was evaluated on the XNLI test set in 15 languages (5010 texts per language, 75150 in total) and the English test sets of [MultiNLI](https://huggingface.co/datasets/multi_nli), [Fever-NLI](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md), [ANLI](https://huggingface.co/datasets/anli), [LingNLI](https://arxiv.org/pdf/2104.07179.pdf) and [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) . Note that multilingual NLI models are capable of classifying NLI texts without receiving NLI training data in the specific language (cross-lingual transfer). This means that the model is also able to do NLI on the other 73 languages mDeBERTa was pre-trained on, but performance is most likely lower than for those languages seen during NLI fine-tuning. The performance on the languages ["bg","el","th"] in the table below is a good indicated of this cross-lingual transfer, as these languages were not included in the training data. 

|XNLI subsets|ar|bg|de|el|en|es|fr|hi|ru|sw|th|tr|ur|vi|zh|
| :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Accuracy|0.794|0.822|0.824|0.809|0.871|0.832|0.823|0.769|0.803|0.746|0.786|0.792|0.744|0.793|0.803|
|Speed (text/sec, A100-GPU)|1344.0|1355.0|1472.0|1149.0|1697.0|1446.0|1278.0|1115.0|1380.0|1463.0|1713.0|1594.0|1189.0|877.0|1887.0|

|English Datasets|mnli_test_m|mnli_test_mm|anli_test|anli_test_r3|fever_test|ling_test|wanli_test|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|Accuracy|0.857|0.856|0.537|0.497|0.761|0.788|0.732|0.794|
|Speed (text/sec, A100-GPU)|1000.0|1009.0|794.0|672.0|374.0|1177.0|1468.0|


Also note that if other multilingual models on the model hub claim performance of around 90% on languages other than English, the authors have most likely made a mistake during testing since non of the latest papers shows a multilingual average performance of more than a few points above 80% on XNLI (see [here](https://arxiv.org/pdf/2111.09543.pdf) or [here](https://arxiv.org/pdf/1911.02116.pdf)). 


## Limitations and bias
Please consult the original DeBERTa-V3 paper and literature on different NLI datasets for potential biases. Moreover, note that the multilingual-nli-26lang-2mil7 dataset was created using machine translation, which reduces the quality of the data for a complex task like NLI. You can inspect the data via the Hugging Face [dataset viewer](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7) for languages you are interested in. Note that grammatical errors introduced by machine translation are less of an issue for zero-shot classification, for which grammar is less important. 


## Citation

If the dataset is useful for you, please cite the following article: 
```
@article{laurer_less_2022,
	title = {Less {Annotating}, {More} {Classifying} – {Addressing} the {Data} {Scarcity} {Issue} of {Supervised} {Machine} {Learning} with {Deep} {Transfer} {Learning} and {BERT} - {NLI}},
	url = {https://osf.io/74b8k},
	language = {en-us},
	urldate = {2022-07-28},
	journal = {Preprint},
	author = {Laurer, Moritz and Atteveldt, Wouter van and Casas, Andreu Salleras and Welbers, Kasper},
	month = jun,
	year = {2022},
	note = {Publisher: Open Science Framework},
}
```


## Ideas for cooperation or questions?
For updates on new models and datasets, follow me on [Twitter](https://twitter.com/MoritzLaurer).
If you have questions or ideas for cooperation, contact me at m{dot}laurer{at}vu{dot}nl or on [LinkedIn](https://www.linkedin.com/in/moritz-laurer/)


## Debugging and issues
Note that DeBERTa-v3 was released in late 2021 and older versions of HF Transformers seem to have issues running the model (e.g. resulting in an issue with the tokenizer). Using Transformers==4.13 or higher might solve some issues. Note that mDeBERTa currently does not support FP16, see here: https://github.com/microsoft/DeBERTa/issues/77



