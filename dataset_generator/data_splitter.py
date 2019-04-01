import random
import os

for filename in os.listdir('JPEGImages'):
        filename = filename.strip('.jpg')

        print('Decididing for ' + filename)

        decider = random.randint(0,100)
        if decider > 85:
            with open('ImageSets/Main/test.txt', 'a') as f:
                f.write(filename + '\n')
        else:
            with open('ImageSets/Main/trainval.txt', 'a') as f:
                f.write(filename + '\n')
            
            if decider > 60:
                with open('ImageSets/Main/val.txt', 'a') as f:
                    f.write(filename + '\n')
            else:
                with open('ImageSets/Main/train.txt', 'a') as f:
                    f.write(filename + '\n')
