f = open("tweetsclass.txt", 'r')
nuevo = open('nuevoTweets.txt','w')
text = f.readline()

for i in range (1, 265):
    try:
        text = f.readline().strip()
        inicio = text.find(';;')
        fin = len(text)
        new_text = text[inicio+2:] +text[:fin]
        inicio = new_text.find(';')
        cleaning_text =  new_text[:inicio]+"    "+text[fin-1]+ '\n'
        nuevo.write(cleaning_text)
    except:
        print("An exception occurred")
f.close()
