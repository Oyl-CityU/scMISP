import os

for alpha in [0.01]: # 0.1, 0.01, 0.001
    for beta in [5]: # 1, 5, 10
        for d in [6]: # 5, 6, 7
            for lambda1 in [0.01]: # 0.001, 0.01, 0.1
                cmd = 'python main.py --alpha {} --beta {} --d {} --lambda1 {}'.format(alpha, beta, d, lambda1)
                print(cmd)
                os.system(cmd)
