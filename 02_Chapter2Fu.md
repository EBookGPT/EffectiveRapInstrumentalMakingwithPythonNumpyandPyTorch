# Chapter 2: Fundamentals of Music Theory

Greetings, my dear students! In the previous chapter, we took the first step in our journey to learn the art of rap instrumental making with Python, Numpy, and PyTorch. We got familiar with the basics of this domain and observed how we can create a beat using pre-selected sounds. 

Now it's time to begin our music theory fundamentals, which is an absolute must for better rap instrumental creation. Without proper knowledge of music theory, a musician is like a sailor without a map in the vast ocean. 

Music theory is not only about raps or beats; it's about music in general. It teaches us about the basic building blocks of musical composition and identifies the patterns in various musical styles. For a budding musician, it is essential to learn the fundamentals of musical theory to create a seamless composition that is appealing to the listener's ear.

The field of music theory can appear daunting at first, but don't worry. In this chapter, we will introduce the fundamental concepts, including scales, chords, and melodies, needed to make rap instrumentals with Python, Numpy, and PyTorch. 

So, fasten your seatbelts and dive in!
# Chapter 2: Fundamentals of Music Theory - Sherlock Holmes Mystery

Sherlock Holmes walked into his living room where Dr. Watson was playing a tune. The detective was usually a fan of classical music but seemed to be quite impressed with Watson's new melody.

"Interesting tune, my dear Watson. Did you compose it yourself?" asked Holmes.

"Yes, I did. I've been creating rap beats in my free time and decided to try out some music theory to create a unique sound," replied Watson.

"Ah, I see. I believe you may have missed a crucial concept in music theory that explains why your tune seems to lack melody," said Holmes.

Watson was surprised by Holmes's statement and curious to learn more about music theory.

"Please explain, Holmes. What have I missed?" asked Watson.

Holmes took a deep breath and began to explain to Watson the concept of scales and how they form the foundation of melodies in music. He explained how different scales convey different emotions and can be used to create a specific atmosphere within a song.

Watson was impressed by Holmes's explanation and didn't realize how much of an impact the scales could have on the final output. Holmes then demonstrated how Watson could use Python and Numpy to create a bassline using a C Major scale.

```python
import numpy as np
import simpleaudio as sa

# Set the fundamental frequency
freq = 261.63

# Set the duration
duration = 2  # seconds

# Generate time array
t = np.linspace(0, duration, duration * 44100, False)

# Generate signal wave
note = np.sin(freq * 2 * np.pi * t)

# Adjust volume
note *= 0.3

# Convert to 16-bit data
note = note * (2 ** 15 - 1) / np.max(np.abs(note))
note = note.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(note, 1, 2, 44100)

# Wait until playback is finished
play_obj.wait_done()
```

By using numpy, Holmes showed Watson how he could create a sine wave with the frequencies of different notes in the C Major scale. Using these notes, Watson could then create a melody that follows the C Major scale, making his tune more cohesive and melodious.

Watson was grateful for Holmes's guidance and decided to implement the same approach in his next song.

# Resolution

Holmes explained to Watson the importance of music theory fundamentals, including the concept of scales. By using Python, Numpy, and simpleaudio libraries, he also demonstrated how Watson could produce a bassline in C Major scale to make his tune more melodious. 

As Watson learned, scales and chords play a vital role in creating melodies that appeal to the audience. With this understanding, Watson could now create rap beats that would resonate with his audience and outdo his past compositions.
Sure, I'd be happy to explain the code!

In the Sherlock Holmes mystery, Watson was struggling to create a melody using music theory fundamentals. Holmes stepped in to explain how scales form the foundation of melodies in music and how different scales can convey different emotions, creating a specific atmosphere in a song. 

Holmes then demonstrated how Python and Numpy could be used to create a basic bassline using the C Major scale. Here's a more in-depth look at the code:

```python
import numpy as np
import simpleaudio as sa

# Set the fundamental frequency
freq = 261.63

# Set the duration
duration = 2  # seconds

# Generate time array
t = np.linspace(0, duration, duration * 44100, False)

# Generate signal wave
note = np.sin(freq * 2 * np.pi * t)

# Adjust volume
note *= 0.3

# Convert to 16-bit data
note = note * (2 ** 15 - 1) / np.max(np.abs(note))
note = note.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(note, 1, 2, 44100)

# Wait until playback is finished
play_obj.wait_done()
```

First, we import the necessary libraries: numpy and simpleaudio. We then proceed to set the fundamental frequency to 261.63 Hz, which represents the note C4. Here, we're actually defining the bassline of our beat. 

Next, we set the duration of the note to 2 seconds. It's essential to define a duration for each note to achieve the required beat tempo.

We then create a time array using numpy's linspace function. The linspace function is used to generate a 1D array with evenly spaced numbers over a specified interval. Here, we're generating an array with 44100 elements, each representing 1/44100th of a second.

Using numpy, we then create a sine wave by multiplying the fundamental frequency, the angle in radians (2*pi), and the time array. Now, we have a basic bassline for our beat.

To adjust the volume of the beat, we multiply the note by 0.3. This step determines how loud the beat will be in the final composition.

Finally, we generate 16-bit sound data compatible with the simpleaudio library. We convert the note to 16-bit data and store it in the variable named 'note'. We then start playback using sa.play_buffer and wait till playback ends with play_obj.wait_done().

In the end, this code snippet is very basic. One can create a more complex and diverse melody using various scales and chords combinations available in music theory. I hope this explanation makes it clear!