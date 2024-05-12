def half(func):
    def imple(val):
        result = func(val)
        return result / 2

    return imple

def multiply(func):
    def inner(val):
        result = func(val)
        return result * result

    return inner

@half
@multiply
def valFunc(val):
    return val

print(valFunc(10))

lamb = lambda x, y: x * y

print(lamb(10, 20))