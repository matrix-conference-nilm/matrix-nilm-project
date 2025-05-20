import torch
import numpy as np
from tqdm import tqdm


def train(
        model,
        train_loader,
        criterion,
        mae_criterion,
        optimizer,
        config,
        epoch,
        training_handler
        ):

    pbar = tqdm(train_loader, desc=f"{config['current_device_name']} Training Epoch {epoch}", dynamic_ncols=True)

    previous_batch = training_handler.get_state_value('batch_count')
    current_batch_count = 0
    running_mse_loss = 0.0
    running_mae_loss = 0.0
    best_mse = np.inf
    model.train()

    for batch in pbar:

        optimizer.zero_grad()
        
        main, device = batch
        main = main.to(config['device']).float()
        device = device.to(config['device']).float()
        
        y_pred = model(main)

        mse_loss = criterion(y_pred, device)

        y_pred_denormalized = (y_pred * config['device_std']) + config['device_mean']
        device_denormalized = (device * config['device_std']) + config['device_mean']
        # for mae denormalize
        mae_loss = mae_criterion(y_pred_denormalized, device_denormalized)

        mse_loss.backward()
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        current_batch_count+=1
        total_batch_count = current_batch_count + previous_batch

        mse_loss.item()

        mse_loss_value = mse_loss.item()
        mae_loss_denomalize = mae_loss.item()
        
        training_handler.update_state_value('train_mse_losses', mse_loss_value)
        training_handler.update_state_value('train_mae_losses', mae_loss_denomalize)
        training_handler.update_state_value('batch_count', total_batch_count)

        running_mse_loss += mse_loss_value
        running_mae_loss += mae_loss_denomalize
        
        avg_mse_loss = running_mse_loss / current_batch_count
        avg_mae_loss = running_mae_loss / current_batch_count

        if total_batch_count % 10 == 0:
            training_handler.save_state()

        if config['debug']:
            if current_batch_count > 10:
                break
        
        if config['wandb']:
            channel = training_handler.get_state_value('channel')
            training_handler.logging(
                    {
                        f"{channel}_{config['current_device_name']}/train_mse" : mse_loss_value,
                        f"{channel}_{config['current_device_name']}/train_mae" : mae_loss_denomalize,
                    }
                )
        else :
            if current_batch_count > 100:
                best_mse = training_handler.current_running_mse
                if avg_mse_loss < best_mse:
                    training_handler.current_running_mse = avg_mse_loss
                if abs(best_mse - avg_mse_loss) < 0.01:
                    training_handler.no_improve += 1
                else :
                    train_loader.no_improve = 0
                # if training_handler.check_early_stopping():
                    # no need to training
                    # break

        pbar.set_description(f"{config['current_device_name']} Training Epoch {epoch}, loss {avg_mse_loss:.3f}, Stop counter {training_handler.no_improve}. {best_mse:.3f}")

    return running_mse_loss / current_batch_count, avg_mse_loss

def validation(
        model,
        val_loader,
        mae_criterion,
        config,
        epoch,
        training_handler
        ):
    
    model.eval()
    pbar = tqdm(val_loader, desc=f"{config['current_device_name']} Validation Epoch {epoch}", dynamic_ncols=True)

    current_batch_count = 0
    running_mae_loss = 0.0
    val_mae_lossed = training_handler.get_state_value('val_mae_losses')
    with torch.no_grad():
        for b in pbar:
        
            main, device = b
            main = main.to(config['device']).float()
            device = device.to(config['device']).float()
            
            y_pred = model(main)

            y_pred_denormalized = (y_pred * config['device_std']) + config['device_mean']
            device_denormalized = (device * config['device_std']) + config['device_mean']
            mae_loss = mae_criterion(y_pred_denormalized, device_denormalized)

            current_batch_count += 1
            running_mae_loss += mae_loss.item()

            avg_mae_loss = running_mae_loss / current_batch_count
            pbar.set_description(f"{config['current_device_name']} Validation Epoch {epoch}, avg loss {avg_mae_loss:.3f}")

            training_handler.update_state_value('val_mae_losses', mae_loss.item())

            if config['debug']:
                if current_batch_count > 10:
                    break

            if config['wandb']:
                channel = training_handler.get_state_value('channel')
                training_handler.logging(
                        {f"{channel}_{config['current_device_name']}/val_mae" : mae_loss.item()}
                    )

        val_best_mae_loss = training_handler.get_state_value('val_best_mae_loss')
        if val_best_mae_loss is None:
            # 첫 번째 결과는 체크포인트 저장
            training_handler.update_state_value('val_best_mae_loss', avg_mae_loss)
            # training_handler.save_model_ckp(model)
        else :
            if avg_mae_loss < val_best_mae_loss:
                training_handler.update_state_value('val_best_mae_loss', avg_mae_loss)
                # training_handler.save_model_ckp(model)
            else :
                training_handler.no_improve += 1
                # if training_handler.check_early_stopping():
                    # training_handler.save_model_ckp(model)

        # save results any way takes to long..
        # 1epoch 훈련시간이 너무 길기 때문에 모든 epoch을 저장
        training_handler.save_state()
        training_handler.save_model_ckp(model)
            
    return running_mae_loss / current_batch_count

