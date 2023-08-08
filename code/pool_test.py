from multiprocessing import Pool
import pickle

def f(x):
	with open('/home/abhishek/pool_test'+str(x)+'.pickle', 'wb') as data:
		pickle.dump([x*x, x**3], data)

if __name__ == '__main__':
    p = Pool(5)
    p.map(f, [1, 2, 3])

'''
with open('/home/abhishek/pool_test2.pickle', 'rb') as data:
		temp=pickle.load(data)

print(temp)
'''