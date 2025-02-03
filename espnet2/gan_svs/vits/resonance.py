import torch
import torch.nn as nn

class AdaptiveLearnableResonance(nn.Module):
    def __init__(self, n_resonances=3, freq_range=(500, 2000)):
        super(AdaptiveLearnableResonance, self).__init__()
        self.n_resonances = n_resonances
        self.freq_range = freq_range

        # 공명 주파수와 Gain은 time_step에 맞게 동적으로 생성
        self.resonance_freqs = None  # 초기값 없음
        self.resonance_gains = None  # 초기값 없음

    def initialize_resonance_params(self, batch_size, time_steps, device):
        # 주파수 초기화 (500Hz ~ 1500Hz)
        self.resonance_freqs = nn.Parameter(
            torch.rand(batch_size, self.n_resonances, time_steps, device=device) * (self.freq_range[1] - self.freq_range[0]) + self.freq_range[0]
        )
        # Gain 초기화 (0~0.1)
        self.resonance_gains = nn.Parameter(
            torch.rand(batch_size, self.n_resonances, time_steps, device=device) * 0.1
        )



    def forward(self, f0):
        """
        Resonance 정보를 반영한 Sinusoidal F0 생성 함수.

        Args:
            f0 (Tensor): 기본 주파수 F0 (B, T, 1)

        Returns:
            Tensor: 공명을 반영한 Sinusoidal F0 (B, 1, T)
        """
        
        if f0.dim() == 3:
            f0 = f0.squeeze(-1)
        else:
            print(f0.shape)      
        batch_size, time_steps = f0.shape

        # 🔥 Batch 크기와 Time Steps 둘 다 체크
        if (self.resonance_freqs is None or 
            self.resonance_freqs.shape[0] != batch_size or  # Batch 크기 체크 추가
            self.resonance_freqs.shape[2] != time_steps):
            self.initialize_resonance_params(batch_size, time_steps, f0.device)
        

        # 시간축 생성 (Pitch에 맞게 동적 조정)
        t = torch.linspace(0, 1, steps=time_steps).to(f0.device).unsqueeze(0).repeat(batch_size, 1)

        # 기본 F0 기반 사인파 생성
        base_wave = torch.sin(2 * torch.pi * f0 * t)

        # 공명 효과 적용
        resonance_wave = 0
        for n in range(self.n_resonances):
            resonance_wave += self.resonance_gains[:, n, :] * torch.sin(2 * torch.pi * self.resonance_freqs[:, n, :] * t)
 
        # 기본 사인파와 Resonance 결합
        combined_wave = base_wave + resonance_wave


        return combined_wave.unsqueeze(1)  # (B, 1, T)
