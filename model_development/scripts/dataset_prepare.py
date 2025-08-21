import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# Converts a string of digits to an integer
def atoi(s):
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n

# Create folder structure for dataset
outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data', outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

# Counters for saved images
angry = disgusted = fearful = happy = sad = surprised = neutral = 0
angry_test = disgusted_test = fearful_test = happy_test = sad_test = surprised_test = neutral_test = 0

# Load FER2013 dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DEV_ROOT = os.path.dirname(SCRIPT_DIR) # scripts'ten bir üst klasöre çıkar (model_development)
RAW_DATA_PATH = os.path.join(MODEL_DEV_ROOT, 'data', 'raw', 'fer2013.csv')

df = pd.read_csv(RAW_DATA_PATH)
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

# Loop through each row in the dataset
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # Convert pixel values to image matrix (48x48)
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # Save image to appropriate train or test folder based on index
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save(f'data/train/angry/im{angry}.png')
            angry += 1
        elif df['emotion'][i] == 1:
            img.save(f'data/train/disgusted/im{disgusted}.png')
            disgusted += 1
        elif df['emotion'][i] == 2:
            img.save(f'data/train/fearful/im{fearful}.png')
            fearful += 1
        elif df['emotion'][i] == 3:
            img.save(f'data/train/happy/im{happy}.png')
            happy += 1
        elif df['emotion'][i] == 4:
            img.save(f'data/train/sad/im{sad}.png')
            sad += 1
        elif df['emotion'][i] == 5:
            img.save(f'data/train/surprised/im{surprised}.png')
            surprised += 1
        elif df['emotion'][i] == 6:
            img.save(f'data/train/neutral/im{neutral}.png')
            neutral += 1
    else:
        if df['emotion'][i] == 0:
            img.save(f'data/test/angry/im{angry_test}.png')
            angry_test += 1
        elif df['emotion'][i] == 1:
            img.save(f'data/test/disgusted/im{disgusted_test}.png')
            disgusted_test += 1
        elif df['emotion'][i] == 2:
            img.save(f'data/test/fearful/im{fearful_test}.png')
            fearful_test += 1
        elif df['emotion'][i] == 3:
            img.save(f'data/test/happy/im{happy_test}.png')
            happy_test += 1
        elif df['emotion'][i] == 4:
            img.save(f'data/test/sad/im{sad_test}.png')
            sad_test += 1
        elif df['emotion'][i] == 5:
            img.save(f'data/test/surprised/im{surprised_test}.png')
            surprised_test += 1
        elif df['emotion'][i] == 6:
            img.save(f'data/test/neutral/im{neutral_test}.png')
            neutral_test += 1

print("Done!")
