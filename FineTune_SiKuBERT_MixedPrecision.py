from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from torch.cuda.amp import autocast, GradScaler






class MaskedDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        labels = self.data.iloc[idx]['Label']  # Danh sách các labels tương ứng với [MASK] tokens

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

        # Tìm vị trí của tất cả các [MASK]
        mask_token_indices = torch.where(input_ids == self.tokenizer.mask_token_id)[0]

        if len(mask_token_indices) == 0:
            raise ValueError(f"Câu số {idx} không có [MASK] token.")

        # Xử lý labels: -100 cho tất cả các vị trí ngoại trừ [MASK]
        labels_tensor = torch.full(input_ids.shape, -100)
        for i, mask_index in enumerate(mask_token_indices):
            if i < len(labels):
                label_token_id = self.tokenizer.convert_tokens_to_ids(labels[i])
                labels_tensor[mask_index] = label_token_id
            else:
                # Nếu số lượng labels ít hơn số [MASK], có thể bỏ qua hoặc xử lý theo cách khác
                pass

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor
        }


def compute_accuracy(predictions, labels):
    """
    Tính accuracy cho các dự đoán tại các vị trí [MASK]
    """
    preds = torch.argmax(predictions, dim=-1)
    mask = labels != -100  # Chỉ xét các vị trí có label
    correct = (preds == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy




def compute_top_k_accuracy(logits, labels, k=5):
    """
    Tính top-k accuracy cho các dự đoán tại các vị trí [MASK].
    """
    top_k = torch.topk(logits, k, dim=-1).indices  # [batch_size, seq_length, k]
    correct = top_k == labels.unsqueeze(-1)  # [batch_size, seq_length, k]
    correct = correct.any(dim=-1)  # [batch_size, seq_length]
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

    # Đọc dataset
    data_path = 'data.csv'
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    print(f"Loaded dataset with {len(df)} samples.")

    # Chuyển đổi cột 'Label' từ chuỗi thành danh sách
    #df['Label'] = df['Label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['Label'] = df['Label'].apply(
        lambda x: x.strip("[]").replace("'", "").split(", ") if isinstance(x, str) else x
    )

    # Kiểm tra và thêm OOV tokens vào tokenizer
    oov_tokens = [token for token in df['Label'].explode().unique() if token not in tokenizer.get_vocab()]
    print(f"Số lượng OOV tokens: {len(oov_tokens)}")

    if oov_tokens:
        tokenizer.add_tokens(oov_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Đã thêm {len(oov_tokens)} OOV tokens vào tokenizer.")
    else:
        print("Tất cả các token đã được bao phủ trong tokenizer.")

    # Khởi tạo K-Fold
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Chuyển đổi dataframe thành numpy array để sử dụng KFold
    data = df.reset_index(drop=True)
    X = data['Text'].values
    y = data['Label'].values  # Không sử dụng trong KFold vì nhiệm vụ là MLM

    # Khởi tạo SummaryWriter cho K-Fold
    writer = SummaryWriter(log_dir='runs/fine_tune_sikubert_kfold')

    # Danh sách lưu trữ metrics cho tất cả các folds
    fold_train_losses = []
    fold_train_accuracies = []
    fold_val_losses = []
    fold_val_accuracies = []
    fold_test_losses = []
    fold_test_accuracies = []

    for fold, (train_ids, val_test_ids) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1} ---")

        # Chia tập val_test thành validation và test
        val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=42)

        # Chia dữ liệu
        train_df = data.iloc[train_ids].reset_index(drop=True)
        val_df = data.iloc[val_ids].reset_index(drop=True)
        test_df = data.iloc[test_ids].reset_index(drop=True)

        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

        # Tạo các Dataset cho tập huấn luyện, validation và test
        train_dataset = MaskedDataset(train_df, tokenizer, max_length=512)
        val_dataset = MaskedDataset(val_df, tokenizer, max_length=512)
        test_dataset = MaskedDataset(test_df, tokenizer, max_length=512)

        # Tạo các DataLoader cho tập huấn luyện, validation và test
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        print(f"Created DataLoader with batch size 16 for Train, Validation, and Test.")

        # Đặt thiết bị
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        print(f"Đang sử dụng thiết bị: {device}")

        # Thiết lập optimizer và scheduler cho từng fold
        optimizer = AdamW(model.parameters(), lr=2e-5)
        print(f"Fold {fold + 1} - Learning rate: {2e-5}")

        epochs = 5
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        scaler = GradScaler()

        # Danh sách lưu trữ loss và accuracy cho fold hiện tại
        fold_train_loss = []
        fold_train_acc = []
        fold_val_loss = []
        fold_val_acc = []
        fold_test_loss = []
        fold_test_acc = []

        # Huấn luyện cho từng fold
        for epoch in range(epochs):
            print(f"\nFold {fold + 1} - Epoch {epoch + 1}/{epochs}")

            # Huấn luyện
            model.train()
            loop = tqdm(train_dataloader, desc=f"Fold {fold +1} Epoch {epoch +1} - Training")
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0



            
            for batch_idx, batch in enumerate(loop):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast():  # Bắt đầu autocast context
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits  # [batch_size, seq_length, vocab_size]

                # Scale loss và thực hiện backward
                scaler.scale(loss).backward()

                # Clip gradients nếu cần thiết
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Thực hiện bước optimizer và scheduler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()

                # Tính accuracy
                accuracy = compute_accuracy(logits, labels)
                epoch_correct += accuracy.item() * input_ids.size(0)
                epoch_total += input_ids.size(0)

                # Lưu trữ metrics
                fold_train_loss.append(loss.item())
                fold_train_acc.append(accuracy.item())

                # Ghi lại loss và accuracy cho mỗi batch
                global_step = fold * epochs * len(train_dataloader) + epoch * len(train_dataloader) + batch_idx
                writer.add_scalar(f'Fold{fold +1}/Loss/train', loss.item(), global_step)
                writer.add_scalar(f'Fold{fold +1}/Accuracy/train', accuracy.item(), global_step)

                loop.set_postfix(loss=loss.item(), accuracy=f"{accuracy.item()*100:.2f}%")







            avg_epoch_loss = epoch_loss / len(train_dataloader)
            avg_epoch_accuracy = epoch_correct / epoch_total
            print(f"Fold {fold +1} - Epoch {epoch +1} Training Loss: {avg_epoch_loss:.4f} | Training Accuracy: {avg_epoch_accuracy*100:.2f}%")

            # Đánh giá trên tập validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                loop = tqdm(val_dataloader, desc=f"Fold {fold +1} Epoch {epoch +1} - Validation")
                for batch_idx, batch in enumerate(loop):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with autocast():  # Bắt đầu autocast context cho đánh giá
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        logits = outputs.logits

                    val_loss += loss.item()

                    accuracy = compute_accuracy(logits, labels)
                    val_correct += accuracy.item() * input_ids.size(0)
                    val_total += input_ids.size(0)

                    # Lưu trữ metrics
                    fold_val_loss.append(loss.item())
                    fold_val_acc.append(accuracy.item())

                    # Ghi lại loss và accuracy cho mỗi batch
                    global_step = fold * epochs * len(train_dataloader) + epoch * len(train_dataloader) + batch_idx
                    writer.add_scalar(f'Fold{fold +1}/Loss/val', loss.item(), global_step)
                    writer.add_scalar(f'Fold{fold +1}/Accuracy/val', accuracy.item(), global_step)

                    loop.set_postfix(loss=loss.item(), accuracy=f"{accuracy.item()*100:.4f}%")



            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_accuracy = val_correct / val_total
            print(f"Fold {fold +1} - Epoch {epoch +1} Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_accuracy*100:.2f}%")

            # Ghi lại loss và accuracy trung bình cho mỗi epoch
            writer.add_scalar(f'Fold{fold +1}/Loss/train_epoch', avg_epoch_loss, epoch)
            writer.add_scalar(f'Fold{fold +1}/Accuracy/train_epoch', avg_epoch_accuracy, epoch)
            writer.add_scalar(f'Fold{fold +1}/Loss/val_epoch', avg_val_loss, epoch)
            writer.add_scalar(f'Fold{fold +1}/Accuracy/val_epoch', avg_val_accuracy, epoch)

        # Đánh giá trên tập test sau khi huấn luyện xong tất cả các epochs cho fold này
        print(f"\nFold {fold +1} - Evaluating on Test Set")
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        
        with torch.no_grad():
            loop = tqdm(test_dataloader, desc=f"Fold {fold +1} - Test Evaluation")
            for batch_idx, batch in enumerate(loop):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast():  # Bắt đầu autocast context cho đánh giá
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                test_loss += loss.item()

                accuracy = compute_accuracy(logits, labels)
                test_correct += accuracy.item() * input_ids.size(0)
                test_total += input_ids.size(0)

                # Lưu trữ metrics
                fold_test_loss.append(loss.item())
                fold_test_acc.append(accuracy.item())

                # Ghi lại loss và accuracy cho mỗi batch
                global_step = fold * epochs * len(train_dataloader) + epoch * len(train_dataloader) + batch_idx
                writer.add_scalar(f'Fold{fold +1}/Loss/test', loss.item(), global_step)
                writer.add_scalar(f'Fold{fold +1}/Accuracy/test', accuracy.item(), global_step)

                loop.set_postfix(loss=loss.item(), accuracy=f"{accuracy.item()*100:.4f}%")



        avg_test_loss = test_loss / len(test_dataloader)
        avg_test_accuracy = test_correct / test_total
        print(f"Fold {fold +1} - Test Loss: {avg_test_loss:.4f} | Test Accuracy: {avg_test_accuracy*100:.2f}%")

        # Lưu kết quả của fold này
        fold_train_losses.append(np.mean(fold_train_loss))
        fold_train_accuracies.append(np.mean(fold_train_acc))
        fold_val_losses.append(np.mean(fold_val_loss))
        fold_val_accuracies.append(np.mean(fold_val_acc))
        fold_test_losses.append(avg_test_loss)
        fold_test_accuracies.append(avg_test_accuracy)

        # Lưu mô hình cho từng fold nếu muốn
        save_model_path = f"extended_sikubert_model_fold{fold +1}"
        tokenizer.save_pretrained(save_model_path)
        model.save_pretrained(save_model_path)
        print(f"Fold {fold +1} - Model và tokenizer đã được lưu tại {save_model_path}")

    # Tính toán và in ra kết quả trung bình và độ lệch chuẩn của các folds
    print("\n=== K-Fold Cross Validation Results ===")
    for fold in range(k_folds):
        print(f"Fold {fold +1} - Train Loss: {fold_train_losses[fold]:.4f}, Train Acc: {fold_train_accuracies[fold]*100:.2f}% | Val Loss: {fold_val_losses[fold]:.4f}, Val Acc: {fold_val_accuracies[fold]*100:.2f}% | Test Loss: {fold_test_losses[fold]:.4f}, Test Acc: {fold_test_accuracies[fold]*100:.2f}%")

    print("\n=== Aggregated Results ===")
    print(f"Average Train Loss: {np.mean(fold_train_losses):.4f} ± {np.std(fold_train_losses):.4f}")
    print(f"Average Train Accuracy: {np.mean(fold_train_accuracies)*100:.2f}% ± {np.std(fold_train_accuracies)*100:.2f}%")
    print(f"Average Validation Loss: {np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}")
    print(f"Average Validation Accuracy: {np.mean(fold_val_accuracies)*100:.2f}% ± {np.std(fold_val_accuracies)*100:.2f}%")
    print(f"Average Test Loss: {np.mean(fold_test_losses):.4f} ± {np.std(fold_test_losses):.4f}")
    print(f"Average Test Accuracy: {np.mean(fold_test_accuracies)*100:.2f}% ± {np.std(fold_test_accuracies)*100:.2f}%")

    # Đóng SummaryWriter
    writer.close()

    save_model_path = f"extended_sikubert_model"
    tokenizer.save_pretrained(save_model_path)
    model.save_pretrained(save_model_path)
    print(f"Fold {fold +1} - Model và tokenizer đã được lưu tại {save_model_path}")

if __name__ == "__main__":
    main()
