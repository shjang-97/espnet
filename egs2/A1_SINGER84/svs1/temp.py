import os
import librosa
import soundfile as sf

# 폴더 경로 설정
input_folder = 'wav_dump2'
output_folder = 'wav_dump'

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# wav_dump 폴더 안의 모든 파일에 대해 반복
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        # 파일 경로 생성
        file_path = os.path.join(input_folder, filename)

        # 오디오 파일 로드
        y, sr = librosa.load(file_path, sr=None)

        # 샘플링 레이트가 44100인지 확인
        if sr == 44100:
            # 샘플링 레이트 변환
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

            # 출력 파일 경로
            output_file_path = os.path.join(output_folder, filename)

            # 변환된 파일 저장
            sf.write(output_file_path, y_resampled, 22050)
            print(f'Saved: {output_file_path}')
        else:
            print(f'Skipped (not 44100 Hz): {file_path}')
