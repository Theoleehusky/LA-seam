from PIL import Image
import numpy as np
from skimage.filters import sobel

title = 'pictures/thing.jpg'
image = np.asarray(Image.open(title)).copy()
width = image.shape[1]
gray = np.asarray(Image.open(title).convert('L'))

for _ in range(width):
    cost = sobel(gray)
    rows, columns = cost.shape
    
    for r in range(rows-2, -1, -1):
        for c in range(columns):
            cost[r][c] += cost[r+1][max(c-1,0):min(columns,c+2)].min()
    
    path = []
    c = cost[0].argmin()
    image[0][c] = [0,0,255]
    path.append(c)
    for r in range(1, rows):
        c = max(c-1,0) + cost[r][max(c-1,0):min(columns,c+2)].argmin()
        image[r][c] = [0,0,255]
        path.append(c)
    
    Image.fromarray(image).save('pictures/thing/' + str(_) + '.png')

    newI = np.empty((rows, columns-1, 3), dtype=np.uint8)
    newG = np.empty((rows, columns-1), dtype=np.uint8)
    for i,j in enumerate(path):
        newI[i][:j] = image[i][:j]
        newI[i][j:] = image[i][j+1:]
        newG[i][:j] = gray[i][:j]
        newG[i][j:] = gray[i][j+1:]
    image = newI
    gray = newG
    
    print(width-_, end=' ')