'''The noise on the frequency of an industrial miniCORI will be measured during 1 minute'''

import time
#import sys
#sys.path.insert( 0 , 'G:/RenD6/PYTHON_LIBRARY/PROPAR/' ) #http://stackoverflow.com/questions/4383571/importing-files-from-different-folder-in-python
import propar
import tkinter as tk 
from tkinter import filedialog

root =tk.Tk()

def write_to_text( file , lijst ):
    '''Saves the list to the text file file
       The data is separated with a, '''
    
    with open( file , 'a' ) as log:
        
        for item in lijst:
            
            if item != lijst[len(lijst)-1]:         # != means is not equal to
                log.write( str( item ) + ','  )
            
            else:
                log.write( str( item ) + '\n' )


status = 0
# Maak een instrument aan
if status == 0:
    COM = input('Select the correct COM and press enter, COM = ')
    comstr = str('com')+str(COM)
    print(comstr)
    print('standard baudrate = 38400')
    baudrate = input('Device baudrate = ')
    print('the default node address is 3')
    node     = input('node = ')
    
    antwoord = input('Are the settings correct? [y/n]')
    if antwoord == 'y':
        status = 1
    else:
        status = 0

if status == 1:
    DUT         = propar.instrument( comport = comstr, address = int(node) , baudrate = int(baudrate) ) # 3 = IFI1, 128 indien direct met comport verbondend
    Serienummer = DUT.readParameter( 92 )

while status ==1:
    meettijd = input('meausurement time in seconds = ')
    root.withdraw()
    print('select the file add .txt for a textfile')
    file_name = filedialog.asksaveasfilename()
    print(file_name)

    # filenaam
    print('filelocation : ' + file_name)

    # header
    header = [ 'tijd' , 'meetwaarde' ]#, 'frequentie' ]
    write_to_text(file_name,header)

    begintijd = time.time()
    teller = 1  
    while time.time() - begintijd < int(meettijd):
        meetwaarde = DUT.readParameter ( 8 )
        #frequentie = DUT.readParameter ( 149 )
        tijd       = round((time.time() - begintijd) , 3 ) # met de functie round wordt de tijd afgerond op
        write_to_text( file_name , [ tijd, meetwaarde ] )
        teller += 1
    print('number of samples = ',teller)
    print('samplespeed = ',round(int(meettijd)/int(teller),4),' s')
    status = 1
