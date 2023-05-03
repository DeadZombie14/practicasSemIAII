#https://stackoverflow.com/questions/65850680/how-to-extract-the-cartesian-coordinates-x-y-of-an-svg-image
from xml.dom import minidom
from svg.path import parse_path
import numpy as np

file = []
doc = minidom.parse('Urban-City-Silhouette.svg')
for ipath, path in enumerate(doc.getElementsByTagName('path')):
    print('Object #', ipath)
    d = path.getAttribute('d')
    parsed = parse_path(d)
    print('Steps:\n', len(parsed), '\n' + '-' * 20)
    for obj in parsed:
        point = ( round(obj.start.imag, 3), round(obj.end.imag, 3) )
        file.append(point)
        print(point)
    #     print(type(obj).__name__, ', start/end coords:', ((round(obj.start.real, 3), round(obj.start.imag, 3)), (round(obj.end.real, 3), round(obj.end.imag, 3))))
    print('-' * 20)
doc.unlink()
np.savetxt('salida.csv', np.array(file), delimiter =',', header='x,y', comments='')