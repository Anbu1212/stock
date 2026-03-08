import os
print("Current directory:", os.getcwd())
print("Files before:", os.listdir('.'))
with open('test.txt', 'w') as f:
    f.write('test')
print("Files after:", os.listdir('.'))
print("Test file saved:", 'test.txt' in os.listdir('.'))
