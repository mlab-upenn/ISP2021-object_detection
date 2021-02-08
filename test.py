class Test:
    def __init__(self):
        self.i = 0
    
    def inc(self, increment):
        self.i += increment
    
    def get_value(self):
        return self.i


blehlist = []

for i in range(10):
    t = Test()
    t.inc(i)
    blehlist.append(t)

for elem in blehlist:
    print(elem.get_value())