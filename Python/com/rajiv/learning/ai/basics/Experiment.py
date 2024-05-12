value = input("Enter a string")
setVal = {"a", "e", "i", "o", "u"}
result = {}
print(''.join(reversed(value)))

for x in value:
    if x in setVal:
        result[x] = result.get(x, 0) + 1

for k, v in result.items():
    print("The count for key", k, "is", v)

