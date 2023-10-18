from subprocess import call
import time


iterations = [1]

iterations = [i for i in range(1, 24)]


f = open('input.txt', 'w')

for _ in range(100):                            # сколько раз посчитать диапазон

    for i in iterations:                      # 0.1-1.0 - то же самое, что (1, 92) шаг - 0.01
        start = time.time()
        print(i)
        input_file = open('input.txt', 'w')
        input_file.write(str(i))
        input_file.close()
        call(['python', 'mirrored_copy.py'])
        
        end = time.time()
        print(end - start)
        
    call(['python', 'graph_output.py'])

