import os

def remove_na():
    FOLDER = "/Users/eb/Desktop/stopsignal_confounds_copy/"
    files = os.listdir(FOLDER)

    for file in files:
        if(file.split(".")[-1]!= "DS_Store"):
            print(FOLDER+file)
            text = open(FOLDER+file, "r")
            text = ''.join([i for i in text])
            text = text.replace("n/a", "0.0")

            x = open(FOLDER+file, "w")
            x.writelines(text)
            x.close()

remove_na()