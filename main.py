import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
#load wavefiles
frequency_sampling, audio_signal = wavfile.read("data/test/te_f_01_01.wav")
audio_signal = audio_signal[:15000]

#extract mfcc featurs
features_mfcc = mfcc(audio_signal, frequency_sampling)
features_mfcc = features_mfcc.T

#extraxt filter bank features
filterbank_features = logfbank(audio_signal, frequency_sampling)

#Print & Display
print("MFCCS :", features_mfcc)
print('Number of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])
print("Filter bank & MFCCs :", filterbank_features)
print('Number of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])
fig, ax = plt.subplots(ncols=2, figsize=(12,4))
ax[0].imshow(features_mfcc)
ax[0].set_title("MFCC")
ax[1].imshow(filterbank_features.T)
ax[1].set_title("Filter bank")
plt.show()

#Save Figure & text
with open('Filterbank&MFCCs_text/test/fm_te_f_01_01.wav.txt', 'w') as f:
    f.write("MFCCs : \n")
    f.write(str(features_mfcc))
    f.write("\n Filter bank : \n")
    f.write(str(filterbank_features))
    f.write("\n Number of windows mfccs :")
    f.write(str(features_mfcc.shape[0]))
    f.write("\n Number of windows Filter bank :")
    f.write(str(filterbank_features.shape[0]))
    f.write("\n Length of each feature mfccs :")
    f.write(str(features_mfcc.shape[1]))
    f.write("\n Length of each feature Filter bank :")
    f.write(str(filterbank_features.shape[1]))

fig.savefig('Filter bank & MFCCs/test/fm_te_f_01_01.wav.png')




