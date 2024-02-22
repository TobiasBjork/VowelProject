import os
from scipy.io import wavfile
def createLanguageFolder(language, vowels):
    # Language as string
    # List of vowels
    
    # Get current folder
    org_folder = os.getcwd()
    # Check if a general language folder exists
    if 'Languages' not in os.listdir():
        os.mkdir('Languages')
    # Change directory
    folder = os.getcwd()+'/Languages'
    os.chdir(folder)
    # Check if specific language folder exists
    if language.upper() not in (name.upper() for name in os.listdir()):
        # If not add folder for specific language
        os.mkdir(language)
        # And a vowel folder
        folder = os.getcwd()+'/'+language
        os.chdir(folder)
        os.mkdir('Vowels')
        # Add vowels
        os.chdir(folder + '/'+ 'Vowels')
        for i in range(0,len(vowels)):
            os.mkdir(vowels[i])

    # Go back to original folder
    os.chdir(org_folder)    

# Vowel folder for each individual person
def updateFolder(language, file, label, id, fs):
    # language - Spoken language as string
    # file - Sound file of vowel in wav format
    # label - Label of vowel as char
    # id - String containing individual information on format XXX

    org_folder = os.getcwd()
    path = org_folder + '/Languages/'+language+'/Vowels/'+label 
    if os.path.exists(path):
        os.chdir(path)
        matches = [name for name in os.listdir() if id in name]
        os.chdir(org_folder)
        path = path + id + label + len(matches)
        wavfile.write(path, fs, file)
    else:
        print('Folder does not exist')


def addVowelToLanguage(language, vowel):
    # Adds vowel to language folder
    org_folder = os.getcwd()
    path = org_folder+'/Languages/'+language+'/Vowels'
    # Checks if language has been added
    if os.path.exists(path):
        # Change directory
        os.chdir(path)
        # Check if vowel not already in set of vowel
        if vowel.upper() not in (name.upper() for name in os.listdir()):
            # Adds vowel to set of vowels for language
            os.mkdir(vowel)
        else:
            print('Vowel already in set of vowels')
        # Go back to original folder
        os.chdir(org_folder)
    else:
        print('Language has not been added')

def removeVowelFromLanguage(language, vowel):
    # Removes vowel from language folder
    path = os.getcwd()+'/Languages/'+language+'/Vowels/'+vowel
    # Checks if vowel has been added
    if os.path.exists(path):
        # Checks if vowel already in set of vowels
        # Removes vowel to set of vowels for language
        os.rmdir(path)
    else:
        print('Issue when removing vowel')

def getListOfVowels(language):
    # Returns list of vowels for language
    vowels = []
    org_folder = os.getcwd()
    path = org_folder+'/Languages/'+language+'/Vowels'
    if os.path.exists(path):
        os.chdir(path)
        vowels = os.listdir()
        os.chdir(org_folder)
    return vowels