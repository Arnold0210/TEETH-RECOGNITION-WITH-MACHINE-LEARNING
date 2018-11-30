from progress.bar import Bar, ChargingBar
import os, time, random
from tqdm import tqdm, trange
with tqdm(total=100, ascii=True) as barra2:
    for num in range(10):
        barra2.update(10)
        time.sleep(0.5)
bar1 = Bar('Procesando:', max=20)
for num in range(20):
    time.sleep(0.2)
    bar1.next()
bar1.finish()


# Declara un objeto de la clase ChargingBar(). Cuando comienza
# el bucle aparece una barra punteada y durante los ciclos los
# puntos "∙" son sustituidos por el carácter "█" hasta alcazar
# el 100%.

bar2 = ChargingBar('Instalando:', max=100)
for num in range(100):
    time.sleep(random.uniform(0, 0.2))
    bar2.next()
bar2.finish()

bar3 = Bar('Escribiendo:', fill='·', suffix='%(percent)d%%')
for i in range(100):
    time.sleep(random.uniform(0, 0.2))
    bar3.next()
bar3.finish()
from progress.spinner import *
import time
spinner = LineSpinner('Leyendo: ')
count = 0
while True:
    count = count + 1
    time.sleep(0.1)
    spinner.next()
    if(count==30):
        break
os.system("PAUSE")



# Declara un objeto de la clase Countdown(). En cada ciclo un
# contador que comienza en 100 va disminuyendo su valor hasta
# alcanzar 0, que marca el fin del bucle.
# El módulo tiene otras clases para declarar objetos
# similares: Counter, Pie y Stack

from progress.counter import Countdown
import time

contador = Countdown("Contador: ")
for num in range(100):
    contador.next()
    time.sleep(0.05)
