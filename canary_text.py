def mid(s, offset, amount):
    return s[offset-1:offset+amount-1]

def generate_canary(text, identifier):
    
    print (binary_string :=  f'{identifier:08b}')
    canary_text = ''
    j=0
    for i, char in enumerate(text):
        if char == ' ':
            if j<=len(binary_string):
                if mid(binary_string, j, 1) == '1':
                    canary_text += '__'
                else:
                    canary_text += '_'  
                j=j+1
            else:
                canary_text += ' '

        else:
            canary_text += char
    
    return canary_text


def detect_version(canary_text):
   
    detected_identifier=''
    maybe= False
    for i, char in enumerate(canary_text):
        if char == '_':
            if maybe == True:

                print (f"{i} - True {detected_identifier}")
                detected_identifier = detected_identifier[:len(detected_identifier)-1] +'1'
               
                maybe = False
            else:
                detected_identifier += '0'
                maybe = True

       


    print (f"{detected_identifier}---")
    
    print(int(detected_identifier, 2))

   

def main():
    # https://chat.openai.com/c/6ab1bb09-5b28-4944-ad5c-67898489b309
    # Example usage
    text = """What is Lorem Ipsum? Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
when an unknown printer took a galley of type and scrambled it to make a type specimen book. 
It has survived not only five centuries, but also the leap into electronic typesetting, 
remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset 
sheets containing Lorem Ipsum passages, and more recently with desktop publishing software 
like Aldus PageMaker including versions of Lorem Ipsum."""

    identifier = 12
    canary = generate_canary(text, identifier)
    print(f"Canary Text for Version {identifier}:\n", canary)

    detect_version(canary)

  

if __name__ == "__main__":
    main()