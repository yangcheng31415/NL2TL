



import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import transformers
import json
import random
import os
import csv
from argparse import ArgumentParser
import numpy as np
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import openai
import torch

# =============================
# 设置 OpenAI API 密钥（请替换为你自己的）
# =============================
openai.api_key = 'Your-API-Key'

def check_semantic_equivalence(pred, target):
    """
    调用 OpenAI API 判断两个时序逻辑语句在语义上是否等价，
    返回一个元组：(是否等价（布尔值）, API 返回的文本)。
    """
    try:
        response = openai.ChatCompletion.create(
            model='o3-mini',
            messages=[
                {"role": "system", "content": "你是一个善于将自然语言转为时序逻辑语句的专家。"},
                {"role": "user", "content": f"请判断下面两个时序逻辑语句在语义上是否等价：\n语句1: {pred}\n语句2: {target}\n请只回答'是'或'否'。"}
            ],
            reasoning_effort='high'
        )
        answer = response.choices[0].message['content'].strip()
        return (answer == '是', answer)
    except Exception as e:
        print("OpenAI API error:", e)
        return (False, "API error")

def correct_parenthe(input_str):
    """
    简单处理括号不匹配问题：如果左括号多，则在末尾补充右括号；如果右括号多，则删除多余部分。
    """
    count = 0
    tokens = input_str.split(' ')
    for index, item in enumerate(tokens):
        if len(item) > 2 and item[-1] == '.':
            tokens[index] = item[:-1]
        if item == '(':
            count += 1
        elif item == ')':
            count -= 1
    if count > 0:
        for i in range(count):
            tokens.append(')')
    elif count < 0:
        for i in range(-count):
            tokens.pop(-1)
    return ' '.join(tokens)

# =============================
# 参数解析
# =============================
parser = ArgumentParser()
parser.add_argument('-seed', '--seed', type=int, default=1203)
parser.add_argument('-name', '--name', default='GLTL')
parser.add_argument('-init_weight', '--init_weight', default='with_pre-train')
parser.add_argument('-data_size', '--data_size', default='0.01-0.09')  # 此处仅作为模式标记
parser.add_argument('-model_checkpoint', '--model_checkpoint', default='t5-large')
args = parser.parse_args()
int_seed = args.seed
dataset_name = args.name
init_weight = args.init_weight
data_size = args.data_size
model_checkpoint = args.model_checkpoint

print(model_checkpoint)
print('*' * 20)
print('\n')

# =============================
# 数据加载与预处理
# =============================
home_path = '/Users/chengyang/Downloads/drive-download-20250309T132839Z-001/Data_transfer_domain/'
if dataset_name == 'GLTL':
    original_list = [
        'GLTL_train_8923_for_transfer_word_midfix.jsonl',
        'GLTL_test_2232_for_transfer_word_midfix.jsonl'
    ]
elif dataset_name == 'navi':
    original_list = ['navi_total_refined.jsonl']
elif dataset_name == 'circuit':
    original_list = ['circuit_total_refined.jsonl']
else:
    print('dataset error!')

dataset_total = []
for file in original_list:
    with open(home_path + file, 'r', encoding='utf-8') as f:
        for line in f:
            dataset_total.append(json.loads(line))
random.shuffle(dataset_total)

csv_file = home_path + '/total_data.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'ltl', 'sentence'])
    for i, entry in enumerate(dataset_total):
        writer.writerow([i, ' '.join(entry['ltl']), ' '.join(entry['sentence'])])

dataset = load_dataset('csv', data_files=csv_file)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "Transform the following sentence into Signal Temporal logic: "
else:
    prefix = ""

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["ltl"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["sentence"] = examples["sentence"]
    model_inputs["ltl"] = examples["ltl"]
    model_inputs["id"] = examples["id"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    count = 0
    for i in range(len(decoded_preds)):
        pred = nltk.sent_tokenize(decoded_preds[i].strip())
        label = nltk.sent_tokenize(decoded_labels[i].strip())
        if pred == label:
            count += 1
    return {'top-1 accuracy': round(count / len(decoded_preds), 6)}

# =============================
# 固定 80%/20% 训练/测试划分
# =============================
train_dataset, test_dataset = dataset['train'].train_test_split(test_size=0.2).values()
# 仅用于最终评估的二次检测，我们只对测试集进行处理
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# =============================
# 模型加载
# =============================
if init_weight == 'with_pre-train':
    model = AutoModelForSeq2SeqLM.from_pretrained('/Users/chengyang/Downloads/checkpoint-62500/')
elif init_weight == 'without_pre-train':
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
else:
    print('Initial model weights error!')

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
model.to(device)

# =============================
# 输出目录设置
# =============================
output_dir_total = '../trained_models/' + dataset_name + '/'
os.makedirs(output_dir_total, exist_ok=True)
output_dir = output_dir_total + dataset_name + '_fixed_split_' + str(int_seed) + '_' + init_weight + '_' + model_checkpoint + '/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

batch_size = 16
model_name_current = model_checkpoint.split("/")[-1] + '-' + dataset_name + "-epoch3"
model_dir = output_dir + model_name_current

args_train = Seq2SeqTrainingArguments(
    output_dir=model_dir,
    run_name=model_name_current,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    seed=int_seed,
    save_total_limit=1,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args_train,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model()

# =============================
# 最终评估：对测试集中的样本进行二次检测以确定最终 eval_top-1 accuracy
# =============================
# 这里设置评估的样本数量，可以设为 1000 或其他你希望的数值，前提是测试集数量足够
num_samples = min(len(tokenized_test_dataset), 1000)
count_correct = 0
for j in range(num_samples):
    inputs = tokenizer([prefix + tokenized_test_dataset[j]['sentence']], max_length=max_input_length, truncation=True, return_tensors="pt").to(device)
    output_generated = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
    decoded_output = tokenizer.batch_decode(output_generated, skip_special_tokens=True)[0].strip()
    predicted_fixed = correct_parenthe(decoded_output)
    target_ltl = tokenized_test_dataset[j]['ltl']
    # 如果预测（经过括号校正后）与目标完全一致，
    # 或者调用 OpenAI API 判断语义等价，则认为该样本正确
    if predicted_fixed == target_ltl or check_semantic_equivalence(predicted_fixed, target_ltl)[0]:
        count_correct += 1

final_accuracy = count_correct / num_samples
print('Final eval_top-1 accuracy on test set:', final_accuracy)
