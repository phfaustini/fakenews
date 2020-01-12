from fileinput import input

def p(): 
     d = {} 
     for line in input('svm.txt'): 
         line = line.rstrip() 
         if line.startswith('Topic'): 
             line = line.split('  ') 
             line = int(line[-1]) 
             if line not in d: 
                 d[line] = 1 
             else: 
                 d[line] += 1 
     return d

print(p())
