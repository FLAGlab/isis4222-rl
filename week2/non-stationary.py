movie = [0,4,2,3,3,3,1,4,2,1,2,3,3,2,2,3,4,4,4,3,1] 
misses = [0,0,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,0,1,1]

n = 20
rewards = [0,[10]*n,[10]*n,[10]*n,[10]*n]
alpha = 0.1

def sum(rewards, i):
    r = 0
    for j in range(0,i):
        r += rewards[j]*alpha*(1-alpha)**(n-j)
    return r

for i in range(1,n):
    m = movie[i]
    if misses[i] == 0:
        rewards[m][i] = 10*(1-alpha)**i + sum(rewards[m], i)
    else:
        inc = i*rewards[m][i-1]%50 + 1
        rewards[m][i-1] += rewards[m][i-1]/inc
        rewards[m][i] = 10*(1-alpha)**i + sum(rewards[m], i)

print(rewards[1][19])
print(rewards[2][19])
print(rewards[3][19])
print(rewards[4][19])
