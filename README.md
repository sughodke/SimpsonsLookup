### Question:
Could I create a Reddit bot which would look for single frames of Simpsons and try to figure out which episode it belongs to?

### Lets go
The Reddit bot part seems straight forward, so I focused on optimizing the Reverse Image lookup.  I found a number of solutions on the web, but decided to roll by own after I noticed the existing solutions did not work so well on animated cartoons.

The basic setup is to create the Reverse lookup database based on keypoints and descriptors found in each frame of an episode.  This information would map scene information to the episode name, air-date and season-episode tuple.  The same descriptor algorithm is computed for the 'needle' frame, the matching descriptor pattern tells us which episode the frame is from.  

### How to Run
Generate Database
```
python CreateSignature.py /Users/SidGhodke/Desktop/vlc-output.ts SimpsonsTest1000.hdf5 --max-frames=1000 --no-video
```

Lookup single frame in the database
```
python FindFrame.py workfile.jpg SimpsonsTest1000.hdf5 
```

Dump/Debug tools
```
python ViewVideo.py 
python -i ORBTest.py 
python Dump.py SimpsonsTest1000.hdf5 | wc -l
```

### Output

Here we can see this process in action, the top is the frame being searched and the bottom is the annotated copy with the descriptors.

![Sample Frame](workfile.jpg)
![Found keypoints/descriptors using ORB algorithm](workfile2.jpg)

### Future Notes

I wanted to try a couple of learning systems, like using the descriptors could I find out who is in the frame.  I.e. Marge, Lisa and Bart are in the frame, so it must be an episode involving them.
