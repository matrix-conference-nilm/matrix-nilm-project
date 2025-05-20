from tqdm import tqdm

import torch
import torch.nn.functional as F

from .metrics import _compute_f1


def train(
        model,
        train_loader,
        optimizer,
        config,
        epoch,
        training_handler
        ):
    
    pbar = tqdm(train_loader, desc=f"{config['current_device_name']} Training Epoch {epoch}", dynamic_ncols=True)

    current_batch_count = 0
    running_loss = 0.0
    running_f1 = 0.0
    avg_loss = 0
    avg_f1 = 0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    model.train()
    for batch in pbar:

        current_batch_count += 1
        optimizer.zero_grad()
        X, y_true = batch[0].float().to(config['device']), batch[1].to(config['device'])
        _, y_pred = model(X)

        loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y_true.float())
        loss.backward()
        optimizer.step()

        y_pred_sigmoid = torch.sigmoid(y_pred).squeeze(-1) > 0.5
        f1, tp, fp, fn, tn, accuracy  = _compute_f1(y_pred_sigmoid, y_true)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        running_loss += loss.item()
        running_f1 += f1.item()

        avg_loss = running_loss / current_batch_count
        avg_f1 = running_f1 / current_batch_count

        pbar.set_description(f"{config['current_device_name']} Training Epoch {epoch}, loss {avg_loss:.2f}, f1 {avg_f1:.3f}, acc {accuracy.item():.2f}")

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    overall_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-8)

    return avg_loss, overall_f1

def valiation(
        model,
        val_loader,
        config,
        epoch,
        training_handler):

    pbar = tqdm(val_loader, desc=f"{config['current_device_name']} Validation Epoch {epoch}", dynamic_ncols=True)

    current_batch_count = 0
    running_loss = 0.0
    running_f1 = 0.0
    avg_loss = 0
    avg_f1 = 0

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for batch in pbar:
            current_batch_count += 1

            X, y_true = batch[0].float().to(config['device']), batch[1].to(config['device'])
            _, y_pred = model(X)

            loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y_true.float())
            y_pred_sigmoid = torch.sigmoid(y_pred).squeeze(-1) > 0.5
            f1, tp, fp, fn, tn, accuracy = _compute_f1(y_pred_sigmoid, y_true)

            running_loss += loss.item()
            running_f1 += f1.item()
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            avg_loss = running_loss / current_batch_count
            avg_f1 = running_f1 / current_batch_count

            pbar.set_description(f"{config['current_device_name']} Validation Epoch {epoch}, loss {avg_loss:.2f}, f1 {avg_f1:.3f}, acc {accuracy.item():.2f}")

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        overall_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-8)

        pbar.set_description(f"{config['current_device_name']} Validation Epoch {epoch}, loss {avg_loss:.2f}, f1 {overall_f1:.3f}, acc {overall_accuracy:.2f}, prec {precision:.2f}, recall {recall:.2f}")

        val_best_mae_loss = training_handler.get_state_value('val_best_mae_loss')
        if val_best_mae_loss is None:
            training_handler.update_state_value('val_best_mae_loss', avg_loss)
            training_handler.update_state_value('val_best_f1', overall_f1.item())
            training_handler.save_model_ckp(model)
        else :
            if avg_loss < val_best_mae_loss:
                training_handler.update_state_value('val_best_mae_loss', avg_loss)
                training_handler.update_state_value('val_best_f1', overall_f1.item())
                training_handler.save_model_ckp(model)

    return avg_loss, overall_f1

                



            