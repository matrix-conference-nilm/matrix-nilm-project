import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# 드롭아웃 비율 설정
hidden_layer_dropout = 0.2

def set_seed():
    """
    난수 생성 시드를 설정하여 실험의 재현성을 보장
    - Python, NumPy, PyTorch, CUDA의 난수 생성기 시드 설정
    - CUDA 연산의 결정론적 동작 보장
    """
    seed = 0
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
def dummy_network(sequence_length, cuda):
    """
    더미 네트워크를 생성하여 Flatten 레이어의 출력 크기를 계산
    Args:
        sequence_length: 입력 시퀀스 길이
        cuda: CUDA 사용 여부
    Returns:
        model: 더미 네트워크 모델
    """
    # Model architecture
    set_seed()
    
    model = torch.nn.Sequential(
        # 첫 번째 컨볼루션 레이어
        torch.nn.Conv1d(out_channels=30, kernel_size=10, in_channels=1, padding='same'),
        torch.nn.ReLU(),

        # 두 번째 컨볼루션 레이어
        torch.nn.Conv1d(out_channels=30, kernel_size=8, in_channels=30, padding='same'),
        torch.nn.ReLU(),

        # 세 번째 컨볼루션 레이어
        torch.nn.Conv1d(out_channels=40, kernel_size=6, in_channels=30, padding='same'),
        torch.nn.ReLU(),

        # 네 번째 컨볼루션 레이어
        torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=40, padding='same'),
        torch.nn.ReLU(), 
        torch.nn.Dropout(p=hidden_layer_dropout),

        # 다섯 번째 컨볼루션 레이어
        torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=50, padding='same'),
        torch.nn.ReLU(), 
        torch.nn.Dropout(p=hidden_layer_dropout),

        # Flatten 레이어
        torch.nn.Flatten(),
    )
    if cuda:
        model.cuda()
    return model

class Seq2Point(nn.Module):
    """
    Seq2Point 모델 클래스
    - 시계열 데이터를 입력으로 받아 단일 포인트 예측을 수행
    - 5개의 컨볼루션 레이어와 2개의 완전 연결 레이어로 구성
    - 배치 정규화와 드롭아웃을 사용하여 과적합 방지
    """
    def __init__(self, sequence_length, cuda):
        """
        Args:
            sequence_length: 입력 시퀀스 길이
            cuda: CUDA 사용 여부
        """
        set_seed()
        super(Seq2Point, self).__init__()
        
        # 더미 네트워크로 Flatten 레이어의 출력 크기 계산
        dummy_model = dummy_network(sequence_length, cuda)
        rand_tensor = torch.randn(1, 1, sequence_length)
        
        if cuda:
            rand_tensor = rand_tensor.to(device='cuda')
        dummy_output = dummy_model(rand_tensor)
        num_of_flattened_neurons = dummy_output.shape[-1]

        # 실제 네트워크 정의
        # 첫 번째 컨볼루션 레이어
        self.conv1 = torch.nn.Conv1d(out_channels=30, kernel_size=10, in_channels=1, padding='same')
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.bn1 = nn.BatchNorm1d(30)  # 배치 정규화

        # 두 번째 컨볼루션 레이어
        self.conv2 = torch.nn.Conv1d(out_channels=30, kernel_size=8, in_channels=30, padding='same')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(30)  # 배치 정규화

        # 세 번째 컨볼루션 레이어
        self.conv3 = torch.nn.Conv1d(out_channels=40, kernel_size=6, in_channels=30, padding='same')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')

        # 네 번째 컨볼루션 레이어
        self.conv4 = torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=40, padding='same')
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')

        # 다섯 번째 컨볼루션 레이어
        self.conv5 = torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=50, padding='same')
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_in', nonlinearity='relu')

        # 완전 연결 레이어
        self.fc1 = torch.nn.Linear(out_features=1024, in_features=num_of_flattened_neurons)
        self.fc2 = torch.nn.Linear(out_features=1, in_features=1024)

        # 드롭아웃 레이어
        self.dropout1 = torch.nn.Dropout(hidden_layer_dropout)
        self.dropout2 = torch.nn.Dropout(hidden_layer_dropout)

        if cuda:
            self.cuda()

    def forward(self, X):
        """
        순전파 연산
        Args:
            X: 입력 텐서 (batch_size, 1, sequence_length)
        Returns:
            x: 출력 텐서 (batch_size, 1)
        """
        # 첫 번째 컨볼루션 블록
        x = self.conv1(X)
        x = F.relu(x)
        x = F.relu(self.bn1(x))

        # 두 번째 컨볼루션 블록
        x = self.conv2(x)
        x = F.relu(x)
        x = F.relu(self.bn2(x))

        # 세 번째 컨볼루션 블록
        x = self.conv3(x)
        x = F.relu(x)

        # 네 번째 컨볼루션 블록
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout1(x)  # 드롭아웃 적용

        # 다섯 번째 컨볼루션 블록
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout2(x)  # 드롭아웃 적용

        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # 완전 연결 레이어
        x = self.fc1(x)
        x = F.relu(x)

        # 출력 레이어
        x = self.fc2(x)
        return x
   