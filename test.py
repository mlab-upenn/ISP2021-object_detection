
d = 100

x = {0: 1, 1: 2, 2:3}
t= sorted([k for k in x.keys() if x[k] is None])

if len(t) > 0:
    print("test")
else:
    x[0] = x[1]
    x[1] = x[2]
    x[2] = d
    print(x)

