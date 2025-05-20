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
    """
    모델 학습 함수
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        criterion: MSE 손실 함수
        mae_criterion: MAE 손실 함수
        optimizer: 최적화기
        config: 설정 정보
        epoch: 현재 에폭
        training_handler: 학습 상태 관리 객체
    Returns:
        running_mse_loss / current_batch_count: 배치당 평균 MSE 손실
        avg_mse_loss: 현재 에폭의 평균 MSE 손실
    """
    # 진행률 표시기 초기화
    pbar = tqdm(train_loader, desc=f"{config['current_device_name']} Training Epoch {epoch}", dynamic_ncols=True)

    # 학습 상태 초기화
    previous_batch = training_handler.get_state_value('batch_count')
    current_batch_count = 0
    running_mse_loss = 0.0
    running_mae_loss = 0.0
    best_mse = np.inf
    model.train()  # 모델을 학습 모드로 설정

    # 배치별 학습
    for batch in pbar:
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 데이터 로드 및 전처리
        main, device = batch
        main = main.to(config['device']).float()
        device = device.to(config['device']).float()
        
        # 순전파 및 손실 계산
        y_pred = model(main)
        mse_loss = criterion(y_pred, device)

        # 역정규화하여 MAE 계산
        y_pred_denormalized = (y_pred * config['device_std']) + config['device_mean']
        device_denormalized = (device * config['device_std']) + config['device_mean']
        mae_loss = mae_criterion(y_pred_denormalized, device_denormalized)

        # 역전파 및 가중치 업데이트
        mse_loss.backward()
        clip_value = 0.5  # 그래디언트 클리핑 값
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        # 배치 카운트 및 손실 업데이트
        current_batch_count += 1
        total_batch_count = current_batch_count + previous_batch

        mse_loss_value = mse_loss.item()
        mae_loss_denomalize = mae_loss.item()
        
        # 학습 상태 업데이트
        training_handler.update_state_value('train_mse_losses', mse_loss_value)
        training_handler.update_state_value('train_mae_losses', mae_loss_denomalize)
        training_handler.update_state_value('batch_count', total_batch_count)

        # 누적 손실 계산
        running_mse_loss += mse_loss_value
        running_mae_loss += mae_loss_denomalize
        
        # 평균 손실 계산
        avg_mse_loss = running_mse_loss / current_batch_count
        avg_mae_loss = running_mae_loss / current_batch_count

        # 10배치마다 상태 저장
        if total_batch_count % 10 == 0:
            training_handler.save_state()

        # 디버그 모드일 때 10배치만 학습
        if config['debug']:
            if current_batch_count > 10:
                break
        
        # wandb 로깅
        if config['wandb']:
            channel = training_handler.get_state_value('channel')
            training_handler.logging(
                    {
                        f"{channel}_{config['current_device_name']}/train_mse" : mse_loss_value,
                        f"{channel}_{config['current_device_name']}/train_mae" : mae_loss_denomalize,
                    }
                )
        else:
            # 조기 종료 체크
            if current_batch_count > 100:
                best_mse = training_handler.current_running_mse
                if avg_mse_loss < best_mse:
                    training_handler.current_running_mse = avg_mse_loss
                if abs(best_mse - avg_mse_loss) < 0.01:
                    training_handler.no_improve += 1
                else:
                    train_loader.no_improve = 0

        # 진행률 표시 업데이트
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
    """
    모델 검증 함수
    Args:
        model: 검증할 모델
        val_loader: 검증 데이터 로더
        mae_criterion: MAE 손실 함수
        config: 설정 정보
        epoch: 현재 에폭
        training_handler: 학습 상태 관리 객체
    Returns:
        running_mae_loss / current_batch_count: 배치당 평균 MAE 손실
    """
    # 모델을 평가 모드로 설정
    model.eval()
    pbar = tqdm(val_loader, desc=f"{config['current_device_name']} Validation Epoch {epoch}", dynamic_ncols=True)

    # 검증 상태 초기화
    current_batch_count = 0
    running_mae_loss = 0.0
    val_mae_lossed = training_handler.get_state_value('val_mae_losses')

    # 그래디언트 계산 없이 검증
    with torch.no_grad():
        for b in pbar:
            # 데이터 로드 및 전처리
            main, device = b
            main = main.to(config['device']).float()
            device = device.to(config['device']).float()
            
            # 순전파 및 손실 계산
            y_pred = model(main)

            # 역정규화하여 MAE 계산
            y_pred_denormalized = (y_pred * config['device_std']) + config['device_mean']
            device_denormalized = (device * config['device_std']) + config['device_mean']
            mae_loss = mae_criterion(y_pred_denormalized, device_denormalized)

            # 배치 카운트 및 손실 업데이트
            current_batch_count += 1
            running_mae_loss += mae_loss.item()

            # 평균 손실 계산
            avg_mae_loss = running_mae_loss / current_batch_count
            pbar.set_description(f"{config['current_device_name']} Validation Epoch {epoch}, avg loss {avg_mae_loss:.3f}")

            # 검증 상태 업데이트
            training_handler.update_state_value('val_mae_losses', mae_loss.item())

            # 디버그 모드일 때 10배치만 검증
            if config['debug']:
                if current_batch_count > 10:
                    break

            # wandb 로깅
            if config['wandb']:
                channel = training_handler.get_state_value('channel')
                training_handler.logging(
                        {f"{channel}_{config['current_device_name']}/val_mae" : mae_loss.item()}
                    )

        # 최고 성능 모델 저장
        val_best_mae_loss = training_handler.get_state_value('val_best_mae_loss')
        if val_best_mae_loss is None:
            # 첫 번째 결과는 체크포인트 저장
            training_handler.update_state_value('val_best_mae_loss', avg_mae_loss)
        else:
            if avg_mae_loss < val_best_mae_loss:
                training_handler.update_state_value('val_best_mae_loss', avg_mae_loss)
            else:
                training_handler.no_improve += 1

        # 모든 에폭의 결과 저장 (학습 시간이 길기 때문에)
        training_handler.save_state()
        training_handler.save_model_ckp(model)
            
    return running_mae_loss / current_batch_count

