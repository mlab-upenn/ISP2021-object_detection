import numpy as np

def foo():
    print(y)

class Test:
    def __init__(self):
        global y
        y = 50

def main():
    test = Test()
    foo()

if __name__ == "__main__":
    main()