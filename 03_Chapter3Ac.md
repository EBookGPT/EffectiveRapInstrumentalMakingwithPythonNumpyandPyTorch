# Chapter 3: Achieving High-Quality Sounds with Digital Signal Processing

In the previous chapter, we had the pleasure of learning about the fundamentals of Music theory with the special guest Andrew Ng. As we draw closer to mastering the techniques of Rap instrumental making, we must now dive into the world of Digital Signal Processing. 

Digital Signal Processing (DSP) is the key to producing high-quality sounds with any music genre, especially Rap. DSP processes sound waves by analysing and performing mathematical operations to improve the quality of the sounds. It has revolutionized the field of music production, allowing musicians and producers to achieve studio-level sound quality without breaking the bank.

In this chapter, we will explore the basics of DSP in the context of Rap production. We will use Python, NumPy, and PyTorch to apply DSP techniques such as filtering, equalizing, and compressing sound files. We will learn how to remove noise from recordings and add effects to make our music sound richer.

So, let us put on our detective hats and get started. Get your headphones ready, and let's dive into the world of DSP for Rap instrumental making!
# Chapter 3: Achieving High-Quality Sounds with Digital Signal Processing

## Sherlock Holmes Mystery: The Muffled Mic

Sherlock Holmes and Dr. John Watson were enjoying a cup of tea when an aspiring rap artist, Jamal, stormed into their office. Jamal had been working tirelessly on his new record, but there was a problem. His microphone had picked up a lot of background noise during the recording, and his voice sounded muffled. Jamal had tried everything he could think of but couldn't fix the problem.

Sherlock Holmes, being a master of sound analysis, offered to help Jamal. Holmes analyzed the recording and quickly realized that the recording had a lot of background noise and a significant part of the signal was distorted. The team set to work to decode the mystery. 

Using Python, NumPy, and PyTorch, Holmes and Watson applied DSP techniques such as filtering and equalization to the sound file. They utilized the Fourier Transform to understand the frequency components of the input signal. They found that Jamal's microphone was picking up a lot of noise due to the ventilation system of the studio. 

To solve the mystery, they put their analysis to the test by applying noise reduction techniques such as the Spectral Subtraction Method and Wavelet Transform to remove the unwanted noise. They also applied compressor effects to make the sound clearer and richer.

## Resolution

After a few hours of filtering and equalizing, the sound file was dramatically improved. The team played the new recording, and Jamal was amazed. The once distorted sound file had become clear, and the background noise was gone, revealing Jamal's voice in all its glory. 

The special guest, Andrew Ng praised the team's analysis and implementation of DSP techniques. He noted how DSP is growing rapidly in the field of music production, and this chapter is a perfect example of the power of DSP in Rap instrumental making.

Jamal was so pleased with the result that he immediately invited the team to produce his new record. And as for Holmes and Watson, they were proud to have cracked another case by using their knowledge of DSP, leaving the world a little bit clearer and rhythmic than before.
## Explanation of the code

In this chapter, we solved the mystery of the muffled mic by using Python, NumPy, and PyTorch to apply various DSP techniques. 

Firstly, we analysed the quality of the sound file by using the Fourier Transform from the NumPy library to understand the frequency components of the input signal. The Fourier Transform breaks down our sound wave into its frequency components and thus allows us to see where the problem lies. The code to perform Fourier transform in Python is as follows:

```python
import numpy as np

# Load the audio file
audio_data, sample_rate = librosa.load("audio_file.wav")

# Perform Fourier Transform
fft_output = np.fft.fft(audio_data)
```

Next, we applied noise reduction using Spectral Subtraction Method and Wavelet Transform. Spectral Subtraction Method allows us to estimate the noise floor of a sound recording and then subtract this from the recording to remove noise. The Wavelet Transform is a technique that analyses a sound recording by breaking it down into different scales and waves. This technique helps to remove unwanted noise from recordings.

```python
import librosa
import numpy as np
import pywt

# Load the audio file
audio_data, sample_rate = librosa.load("audio_file.wav")

# Perform Spectral Subtraction for noise reduction
n_fft = int(0.025 * sample_rate)
hop_length = int(0.010 * sample_rate)
window = np.hanning(n_fft)
noise_stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length, window=window)
noise_stft_mean = noise_stft.mean(axis=1)
audio_data_stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length, window=window)
audio_data_stft_subtracted = audio_data_stft - noise_stft_mean.reshape((-1,1))

# Perform Wavelet Transform for noise reduction
coeffs = pywt.wavedec(audio_data, 'db2', level=1)
sigma = (1 / 0.6745) * mad(coeffs[-1])
threshold = sigma * np.sqrt(2 * np.log(len(audio_data)))
coeffs_thresholded = [pywt.threshold(i, threshold) for i in coeffs]
audio_data_wiener = pywt.waverec(coeffs_thresholded, 'db2')
```

Finally, we applied compression effects to improve the sound quality further. To apply compression on the sound file, we used PyTorch's Sigmoid activation function. Sigmoid application helps to make the sound richer and helps to reduce any unwanted distortions in the recording.

```python
import librosa
import numpy as np
import torch

# Load the audio file
audio_data, sample_rate = librosa.load("audio_file.wav")

# Normalize the input signal
audio_data_normalized = librosa.util.normalize(audio_data)

# Apply compression with PyTorch
tensor_audio = torch.Tensor(audio_data_normalized)
tensor_audio_compressed = torch.sigmoid(tensor_audio)

# Convert tensor to numpy array
audio_data_compressed = tensor_audio_compressed.numpy()
```

By combining these techniques, we were able to solve the mystery of the muffled mic and improve the sound quality of our recording. Remember that DSP techniques can significantly improve the quality of any sound recording, and using Python, NumPy, and PyTorch for DSP is an incredibly powerful tool for any music producer.