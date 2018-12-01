import numpy as np
tabla = [[1, 2, 3], [4, 5, 6], [7, 8, 9],[8,9,0]]
header = []
label=[['label1'],['label2'],['label3']]
print(len(np.array(tabla).T[0]))
'''with open('tabla.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(tabla)
extract, countPixel = ext.extract(src)
colors_image = []
for colors in extract:
    for l in colors[0]:
        colors_image.append((l))
matrix_data.append(colors_image)'''