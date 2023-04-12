# Chapter 5: Rhythm Construction using Beat Slicing and Sequencing 

Welcome to the fifth chapter of our book on Effective Rap Instrumental Making with Python, Numpy and PyTorch. In the previous chapter, we learned about Sample Collection and Audio manipulation with NumPy. We also had a special guest, TOKiMONSTA, who shared her experience in making beats and using audio manipulation techniques. 

In this chapter, we will continue our journey and discover the art of constructing rhythm using beat slicing and sequencing. Rhythm is an integral part of music, and it is particularly important in rap instrumentals. A good rhythm keeps the song moving, and it can make the difference between a mediocre beat and a great one.

We will explore how to extract rhythmic elements from audio samples, slice and rearrange them into new patterns, and sequence them to create compelling beats. We will use Python, Numpy, and PyTorch to implement various techniques that can help us in this endeavor.

Get ready to dive into the world of rhythm construction and create the most effective rap instrumentals with our techniques. Let's get started!
# Chapter 5: Rhythm Construction using Beat Slicing and Sequencing

Welcome to the fifth chapter of our book on Effective Rap Instrumental Making with Python, Numpy, and PyTorch. In this chapter, we will be following the famous detective Sherlock Holmes as he solves a mystery related to beat slicing and sequencing. 

## The Case of the Missing Rhythm

Sherlock Holmes was sitting in his apartment when there was a knock on the door. It was his dear friend and collaborator, TOKiMONSTA, a talented musician who also used Python and technology to create music. She was clearly distraught and spoke in a hurried manner, "Sherlock, I need your help! I have been working on this new rap instrumental for days, but the rhythm just doesn't sound right. I have tried everything I know, but something is off. Can you help me figure out what's wrong?"

Holmes was intrigued and agreed to help. He asked TOKiMONSTA to play the instrumental, and he listened carefully. The beat had a good melody and sound, but there was definitely something missing. Holmes asked to see the code, and they went over it together. They noticed that some of the beats were not lining up, and there were gaps in the rhythm.

Holmes knew that to fix the rhythm, they needed to focus on the beat slicing and sequencing. He asked TOKiMONSTA to provide him with some audio samples that could be used to create new rhythms. Together, they used NumPy to slice and rearrange the beats, adding some additional sound effects to create new audio patterns. They then used PyTorch to sequence these patterns together to create a rhythm that was catchy and effective.

After some experimentation, they were able to create a new rhythm that sounded great. They added this new rhythm to the instrumental and played it again. This time, the beat sounded much better, and TOKiMONSTA was thrilled. She said, "Sherlock, you truly are a genius! You were able to solve the mystery of the missing rhythm!"

## The Resolution

Through their work with beat slicing and sequencing, Holmes and TOKiMONSTA were able to unlock the key to a great rhythm. By using Python, NumPy, and PyTorch, they were able to extract meaningful beats, slice and rearrange them into new patterns, and sequence them together to create a compelling rhythm that elevated the instrumental to new heights. 

They both knew that working with rhythm requires careful attention to detail, creativity, and ingenuity. By working together, Holmes and TOKiMONSTA were able to create a beat that not only sounded good but also had a unique flavor. Through their work, they discovered that sometimes the answer to a musical mystery lies in the precise slicing and sequencing of beats.
To solve the mystery of the missing rhythm, Sherlock Holmes and TOKiMONSTA used Python, NumPy, and PyTorch to slice and sequence audio samples. Let’s take a closer look at the code they used.

First, they imported the necessary libraries:

```python
import numpy as np
import torch
import librosa
```

Then, they loaded an audio sample using Librosa:

```python
audio_path = 'sample_audio.wav'
y, sr = librosa.load(audio_path)
```

Next, they converted the audio sample to a spectrogram using a short-time Fourier transform.

```python
n_fft = 2048
hop_length = 512
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
```

Then, they used NumPy to slice the audio sample into smaller beat-sized clips:

```python
win_len = 1024
n_windows = int(np.ceil(S.shape[1] / win_len))
S_pad = np.pad(S, [(0, 0), (0, n_windows * win_len - S.shape[1])], mode='constant')
windows = np.vstack(np.split(S_pad, n_windows, axis=1))
```
The `win_len` variable determines the length of each beat-sized clip, which is then split into non-overlapping windows.

Once they had all the beat-sized clips extracted and stored in a NumPy array. They could then slice and rearrange the beats in unique ways to create new audio patterns. Here’s an example of how they shuffled the beats:

```python
np.random.shuffle(windows)
```

After rearranging the beats, it was time to sequence them together to create a rhythm. They used PyTorch’s `LSTM` module to do this:

```python
n_input = windows.shape[1]
n_hidden = 64
n_layers = 2
n_output = n_input
model = torch.nn.LSTM(n_input, n_hidden, n_layers, batch_first=True)
out = model(torch.Tensor(windows[np.newaxis, :, :]))
y = out[0][0].detach().numpy()
```

The `LSTM` module takes in a sequence of beat-sized clips and outputs a new sequence based on the patterns it has learned. Finally, they combined the new sequence of beat-sized clips into a single audio clip using an inverse short-time Fourier transform, and saved the result:

```python
S_out = librosa.istft(y)
y_out = librosa.util.normalize(S_out)
audio_out_path = 'new_audio.wav'
librosa.output.write_wav(audio_out_path, y_out, sr=sr)
```

And that's how Sherlock Holmes and TOKiMONSTA solved the mystery of the missing rhythm using Python, NumPy, and PyTorch!