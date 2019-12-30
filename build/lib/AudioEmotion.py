import librosa
import soundfile
import numpy as np
import os
import pickle
import glob
import pyaudio
import wave
import time
from sys import byteorder
from array import array
from struct import pack
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

rate = 16000
threshold = 500
chunk_size = 1024
form = pyaudio.paInt16
silence_threshold = 30


def extraitAudio(fich, mfcc, chroma, mel):
    with soundfile.SoundFile(fich, 'r') as s:
        readS = s.read(dtype="float32")
        rateSample = s.samplerate
        res = np.array([])
        if chroma:
            stft = np.abs(librosa.stft(readS))
        if mfcc:
            m = np.mean(librosa.feature.mfcc(y=readS, sr=rateSample, n_mfcc=40).T, axis=0)
            res = np.hstack((res, m))
        if chroma:
            chromaStft = np.mean(librosa.feature.chroma_stft(sr=rateSample, S=stft).T, axis=0)
            res = np.hstack((res, chromaStft))
        if mel:
            mels = np.mean(librosa.feature.melspectrogram(y=readS, sr=rateSample).T, axis=0)
            res = np.hstack((res, mels))
    return res


dictEmo = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

observedEmo = ["calm", "happy", "fear", "angry"]


def dataLoader(test_size=0.2):
    x = list()
    y = list()
    for fich in glob.glob("speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        f_name = os.path.basename(fich)
        emo = dictEmo[f_name.split("-")[2]]
        if emo not in observedEmo:
            continue
        ex = extraitAudio(fich, mfcc=True, chroma=True, mel=True)
        x.append(ex)
        y.append(emo)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


def model_trainer(alpha=0.01):
    if not os.path.isdir("ModelResult") and not os.path.isfile("//ModelResult//model.pickle"):
        print("Training and Testing Model !!! Please Wait....")
        x_train, x_test, y_train, y_test = dataLoader(test_size=0.2)

        print((x_train.shape[0], x_test.shape[0]))
        print(f'Extracted Audio: {x_train.shape[1]}')

        model = MLPClassifier(alpha=alpha, batch_size=256, epsilon=1e-08,
                              hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_predict)
        print("Finished...")
        print("Accuracy: {:.2f}%".format(accuracy*100))

        os.mkdir("ModelResult")
        trainedModel = (model, accuracy)
        pickle.dump(trainedModel, open("ModelResult/model.pickle", "wb"))
    return trainedModel


def load_model():
    if os.path.isdir("ModelResult") and os.path.isfile("ModelResult/model.pickle"):
        trainedModel = pickle.load(open("ModelResult/model.pickle", "rb"))
    return trainedModel


def normalize_sound(sound_data):
    vol_max = 16384
    avg = float(vol_max) / max(abs(i) for i in sound_data)

    arr = array('h')
    for i in sound_data:
        arr.append(int(i * avg))
    return arr


def trim_sound(sound_data):
    def _trim_sound(sound_data):
        is_started = False
        arr = array('h')

        for i in sound_data:
            if not is_started and abs(i) > threshold:
                is_started = True
                arr.append(i)
            elif is_started:
                arr.append(i)
        return arr

    # Trim to the left
    sound_data = _trim_sound(sound_data)

    # Trim to the right
    sound_data.reverse()
    sound_data = _trim_sound(sound_data)
    sound_data.reverse()

    return sound_data


def add_silence(sound_data, seconds):
    arr = array('h', [0 for i in range(int(seconds * rate))])
    arr.extend(sound_data)
    arr.extend([0 for i in range(int(seconds * rate))])
    return arr


def is_silent(sound_date):
    return max(sound_date) < threshold


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=form, channels=1, rate=rate, input=True, output=True,
                    frames_per_buffer=chunk_size)

    silent_numb = 0
    is_started = False
    arr = array('h')
    now = time.time() + 5

    while 1:
        sound_data = array('h', stream.read(chunk_size))
        if byteorder == 'big':
            sound_data.byteswap()
        arr.extend(sound_data)

        silent = is_silent(sound_data)

        if silent and is_started:
            silent_numb += 1
        elif not silent and not is_started:
            is_started = True

        if is_started and silent_numb > silence_threshold:
            break
        elif is_started and time.time() > now:
            break

    sample_width = p.get_sample_size(form)
    stream.stop_stream()
    stream.close()
    p.terminate()

    arr = normalize_sound(arr)
    arr = trim_sound(arr)
    arr = add_silence(arr, 0.5)

    return sample_width, arr


def record_to_file(path):
    sample_width, data = record_audio()
    data = pack('<' + ('h' * len(data)), *data)

    w = wave.open(path, 'wb')
    w.setnchannels(1)
    w.setframerate(rate)
    w.setsampwidth(sample_width)
    w.writeframes(data)
    w.close()

def main_menu():
    time.sleep(2)
    print("\n*******Main Menu*******\n ")
    print("1 - Train the Model.")
    print("2 - Check the accuracy of the Model.")
    print("3 - Record your voice.")
    print("4 - Exit.")
    print("What you want to do ? ")
    rep = int(input("Choose a Number : "))
    return rep




if __name__ == '__main__':
    try:
        rep = main_menu()
        while rep != 4:
            if rep == 1:
                if not os.path.isdir("ModelResult") and not os.path.isfile("//ModelResult//model.pickle"):
                    al = float(input("Choose the value of alpha, a float number from 0.0001 to 0.1"
                                     "\nrecommended 0.01 for better Model Accuracy : "))
                    trainedModel = model_trainer(al)
                    rep = main_menu()
                else:
                    print("Model already exists, Delete ? (y/n)")
                    rep1 = input()
                    if rep1.lower() == "y":
                        os.system('rd /S ModelResult')
                        print("Model deleted")
                    elif rep1.lower() == "n":
                        print("Going back to main menu...")
                    else:
                        print("choose the right answer...")
                    rep = main_menu()

            elif rep == 2:
                if os.path.isdir("ModelResult") and os.path.isfile("ModelResult/model.pickle"):
                    trainedModel = load_model()
                    print("Accuracy: {:.2f}%".format(trainedModel[1] * 100))
                    rep = main_menu()
                else:
                    print("Model doesn't exist")
                    rep = main_menu()
            elif rep == 3:
                if not os.path.isfile("ModelResult/model.pickle"):
                    print("Model doesn't exist !! Train the Model first.")
                    rep = main_menu()
                else:
                    trainedModel = load_model()
                    print("Recording Sound, TALK PLEASE!!! ...")
                    file = "recorded_Audio.wav"
                    record_to_file(file)
                    ex = extraitAudio(file, mfcc=True, chroma=True, mel=True).reshape(1, -1)
                    res = trainedModel[0].predict(ex)[0]
                    print(res)
                    rep = main_menu()
            else:
                print("Choose a correct number PLEASE!!!")
                rep = int(input("Choose a Number : "))
    except:
        print("Erreur")






# trainedModel = model_trainer()
# print("Accuracy: {:.2f}%".format(trainedModel[1]*100))
# print("Recording Sound, TALK PLEASE!!! ...")
# file = "recorded_Audio.wav"
# record_to_file(file)
# ex = extraitAudio(file, mfcc=True, chroma=True, mel=True).reshape(1, -1)
# res = trainedModel[0].predict(ex)[0]
# print(res)

# root = ThemedTk(theme="equilux")
# root.geometry("350x350")
# # ttk.Style().configure("TButton", padding=6, relief="flat", background="#ccc")
#
#
#
#
# res = ""
# isClicked = False
# def startRecord():
#     now = time.time() + 5
#
#     label1.configure(text="Talk Please...")
#
#     file = "recorded_Audio.wav"
#     record_to_file(file)
#     ex = extraitAudio(file, mfcc=True, chroma=True, mel=True).reshape(1, -1)
#     res = trainedModel[0].predict(ex)[0]
#     isClicked = True
#
#     return res
#
#
# def deleteModel():
#     os.system("rm -rf /ModelResult")
#
# def trainModel():
#
#
# btn = ttk.Button(text="Sample", command=deleteModel)
# btn.pack()
#
# root.mainloop()
#
# # bt = tkinter.Button(window, text="Click to Record", command=clicked)
# # bt.grid(column=1, row=0)
#
# # window.geometry('350x200')
# # window.mainloop()
