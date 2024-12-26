from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Định nghĩa lớp MaskedDataset
class MaskedDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        label = self.data.iloc[idx]['Label']

        # Tokenize văn bản
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()  # [max_length]
        attention_mask = encoding['attention_mask'].squeeze()  # [max_length]

        # Tìm vị trí của [MASK]
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[0]

        if len(mask_token_index) != 1:
            raise ValueError(f"Câu số {idx} không có đúng một [MASK] token.")

        mask_index = mask_token_index.item()

        # Chuyển đổi label thành token ID
        label_id = self.tokenizer.convert_tokens_to_ids(label)

        # Tạo labels: -100 cho tất cả các vị trí ngoại trừ [MASK]
        labels = torch.full(input_ids.shape, -100)
        labels[mask_index] = label_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Hàm tính accuracy
def compute_accuracy(predictions, labels):
    """
    Tính accuracy cho các dự đoán tại vị trí [MASK]
    """
    preds = torch.argmax(predictions, dim=-1)
    mask = labels != -100  # Chỉ xét các vị trí có label
    correct = (preds == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy

# Hàm tính top-k accuracy
def compute_top_k_accuracy(logits, labels, k=5):
    """
    Tính top-k accuracy cho các dự đoán tại vị trí [MASK].
    """
    top_k = torch.topk(logits, k, dim=-1).indices
    correct = top_k == labels.unsqueeze(-1)
    correct = correct.any(dim=-1)
    accuracy = correct.sum().float() / (labels != -100).sum().float()
    return accuracy

# Cài đặt và khởi tạo
def main():
    sikubert_model_path = "SIKU-BERT/sikubert"

    tokenizer = BertTokenizer.from_pretrained(sikubert_model_path)
    model = BertForMaskedLM.from_pretrained(sikubert_model_path)

    nom_vocab_file = 'vocab_Han_Nom.txt'
    with open(nom_vocab_file, 'r', encoding='utf-8') as f:
        nom_vocab = [line.strip() for line in f.readlines()]

    # Kiểm tra xem các token đã tồn tại hay chưa trước khi thêm
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in nom_vocab if token not in existing_tokens]

    num_added_tokens = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Đã thêm {num_added_tokens} tokens vào tokenizer.")

    sample_sentences = [
        "𠄎𠂤𡿨𡯨",
        "民浪屡奴群低",
        "戈䀡󰘚倍踈兮㐌仃"
    ]

    for sentence in sample_sentences:
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"Câu: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print("-" * 50)

    # Đọc dataset
    data_path = 'masked_dataset.csv'
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    print(f"Loaded dataset with {len(df)} samples.")

    # Kiểm tra và thêm OOV tokens vào tokenizer
    oov_tokens = [token for token in df['Label'].unique() if token not in tokenizer.get_vocab()]
    print(f"Số lượng OOV tokens: {len(oov_tokens)}")

    if oov_tokens:
        tokenizer.add_tokens(oov_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Đã thêm {len(oov_tokens)} OOV tokens vào tokenizer.")
    else:
        print("Tất cả các token đã được bao phủ trong tokenizer.")

    # Chia dữ liệu thành tập huấn luyện và validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Tạo các Dataset cho tập huấn luyện và validation
    train_dataset = MaskedDataset(train_df, tokenizer, max_length=512)
    val_dataset = MaskedDataset(val_df, tokenizer, max_length=512)

    # Tạo các DataLoader cho tập huấn luyện và validation
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(f"Created DataLoader with batch size 16.")

    # Đặt thiết bị
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f"Đang sử dụng thiết bị: {device}")

    # Thiết lập optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)  # Tăng learning rate từ 1e-6 thành 2e-5
    print(f"Learning rate: {2e-5}")

    # Khởi tạo Scheduler
    total_steps = len(train_dataloader) * 5  # epochs=5
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Khởi tạo SummaryWriter
    writer = SummaryWriter(log_dir='runs/fine_tune_sikubert')

    epochs = 5

    # Danh sách lưu trữ loss và accuracy
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Huấn luyện
    for epoch in range(epochs):
        # Huấn luyện
        model.train()
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, batch in enumerate(loop):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits  # [batch_size, seq_length, vocab_size]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            # Tính accuracy
            accuracy = compute_accuracy(logits, labels)
            epoch_correct += accuracy.item() * input_ids.size(0)
            epoch_total += input_ids.size(0)

            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())

            # Ghi lại loss và accuracy cho mỗi batch
            global_step = epoch * len(train_dataloader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Accuracy/train', accuracy.item(), global_step)

            loop.set_postfix(loss=loss.item(), accuracy=f"{accuracy.item()*100:.2f}%")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_accuracy = epoch_correct / epoch_total
        print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f} | Training Accuracy: {avg_epoch_accuracy*100:.2f}%")

        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()

                accuracy = compute_accuracy(logits, labels)
                val_correct += accuracy.item() * input_ids.size(0)
                val_total += input_ids.size(0)

                val_losses.append(loss.item())
                val_accuracies.append(accuracy.item())

                loop.set_postfix(loss=loss.item(), accuracy=f"{accuracy.item()*100:.2f}%")

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_accuracy = val_correct / val_total
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_accuracy*100:.2f}%")

        # Ghi lại loss và accuracy trung bình cho mỗi epoch
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', avg_epoch_accuracy, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val_epoch', avg_val_accuracy, epoch)

    # Đóng SummaryWriter
    writer.close()

    # Lưu model và tokenizer
    save_model_path = "extended_sikubert_model"
    tokenizer.save_pretrained(save_model_path)
    model.save_pretrained(save_model_path)
    print(f"Model và tokenizer đã được lưu tại {save_model_path}")

    # Dự đoán và đánh giá
    test_sentences = [
        "𨤧𡗶坦常欺[MASK]𡏧",
        "客𦟐[MASK]𡗉餒迍邅",
        "撑[MASK]𠽉瀋層𨕭"
    ]

    model.eval()
    with torch.no_grad():
        for i, sentence in enumerate(test_sentences):
            encoding = tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, seq_length, vocab_size]

            # Tìm vị trí của [MASK]
            mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

            if len(mask_token_index) != 1:
                raise ValueError(f"Câu: {sentence} không có đúng một [MASK] token.")

            mask_index = mask_token_index.item()

            # Lấy dự đoán tại vị trí [MASK]
            predicted_token_id = logits[0, mask_index, :].argmax(dim=-1)
            predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

            # Lấy label thực tế từ dataset nếu có
            actual_label = df.iloc[i]['Label'] if i < len(df) else None

            print(f"Câu: {sentence}")
            print(f"Predicted Label: {predicted_token}")
            if actual_label:
                print(f"Actual Label: {actual_label}")
                print(f"Dự đoán đúng không? {'Có' if predicted_token == actual_label else 'Không'}")
            print("-" * 50)

if __name__ == "__main__":
    main()
