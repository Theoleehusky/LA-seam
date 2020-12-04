from PIL import Image
import numpy as np
from skimage import filters

title = 'pictures/interim.jpg'
image = np.asarray(Image.open(title)).copy()
width = image.shape[1]
gray = np.dot(image, [0.299, 0.587, 0.114]).astype(np.uint8)

for _ in range(width-1):
    cost = filters.sobel(gray)
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
    
    Image.fromarray(image).save('pictures/interim/' + str(_) + '.png')
    
    columns = image.shape[1]-1
    newI = np.empty((rows, columns, 3), dtype=np.uint8)
    newG = np.empty((rows, columns), dtype=np.uint8)
    for i,j in enumerate(path):
        newI[i][:j] = image[i][:j]
        newI[i][j:] = image[i][j+1:]
        newG[i][:j] = gray[i][:j]
        newG[i][j:] = gray[i][j+1:]
    image = newI
    gray = newG
    
    print(width-_-1, end=' ')