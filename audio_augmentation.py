import librosa
import numpy as np
import os
from hparams import hparams
import soundfile as sf
from set_lists import *


def rnd_numbers_list():
    length = hparams.aug_number
    number = np.random.randint(0,9)
    number_l = [number]
    while len(number_l) < length:
        number = np.random.randint(0, 9)
        while number in number_l:
            number = np.random.randint(0, 9)
        number_l.append(number)
    return number_l

def get_genre(hparams):
    return hparams.genres

def load_list(list_name, hparams):
    with open(os.path.join(hparams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()

    return file_names

def get_item(hparams, genre):
    print(os.path.join(hparams.genres_path,str(genre)))
    return librosa.util.find_files(os.path.join(hparams.genres_path,str(genre)))


def readfile(file_name, hparams):
    y, sr = librosa.load(file_name, hparams.sample_rate)
    return y, sr


def change_pitch_and_speed(data):
    y_pitch_speed = data.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed


def change_pitch(data, sr):
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def value_aug(data):
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    y_aug = y_aug * dyn_change
    return y_aug


def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise


def hpss(data):
    y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'))
    return y_harmonic, y_percussive


def shift(data):
    return np.roll(data, 1600)


def stretch(data, rate=1):
    input_length = len(data)
    streching = librosa.effects.time_stretch(data, rate)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

def change_speed(data):
    y_speed = data.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def main():
    print('Augmentation')

    genres = get_genre(hparams)
    list_names = ['train_list.txt']
    for list_name in list_names:
        file_names = load_list(list_name, hparams)
        with open(os.path.join(hparams.dataset_path, list_name),'w') as f:
            for i in file_names:
                f.writelines(i+'\n')
                f.writelines(i.replace('.wav', 'a.wav' + '\n'))
                f.writelines(i.replace('.wav', 'b.wav' + '\n'))
                f.writelines(i.replace('.wav', 'c.wav' + '\n'))
                f.writelines(i.replace('.wav', 'd.wav' + '\n'))
                f.writelines(i.replace('.wav', 'e.wav' + '\n'))
                f.writelines(i.replace('.wav', 'f.wav' + '\n'))
                f.writelines(i.replace('.wav', 'g.wav' + '\n'))
                f.writelines(i.replace('.wav', 'h.wav' + '\n'))
                f.writelines(i.replace('.wav', 'i.wav' + '\n'))
    print(genres)
    for genre in genres:
        item_list = get_item(hparams, genre)
        print(item_list)
        for file_name in item_list:
            print(file_name)
            y, sr = readfile(file_name, hparams)
            data_noise = add_noise(y)
            data_roll = shift(y)
            data_stretch = stretch(y)
            pitch_speed = change_pitch_and_speed(y)
            pitch = change_pitch(y, hparams.sample_rate)
            speed = change_speed(y)
            value = value_aug(y)
            y_harmonic, y_percussive = hpss(y)
            y_shift = shift(y)

            save_path = os.path.join(file_name.split(genre + '.')[0])
            save_name =  genre + '.'+file_name.split(genre + '.')[1]
            print(save_name)

            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'a.wav')), data=data_noise,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'b.wav')), data=data_roll,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'c.wav')), data=data_stretch,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'd.wav')), data=pitch_speed,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'e.wav')), data=pitch,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'f.wav')), data=speed,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'g.wav')), data=value,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'h.wav')), data=y_percussive,
                                     samplerate=hparams.sample_rate)
            sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'i.wav')), data=y_shift,
                                     samplerate=hparams.sample_rate)
        print('finished')

def main_reduced():
    print('Reduced Augmentation')

    genres = get_genre(hparams)
    list_names = ['train_list.txt']
    for list_name in list_names:
        file_names = load_list(list_name, hparams)#train list
        with open(os.path.join(hparams.dataset_path, str(list_name)+"_aug"),'w') as f:
            for i in file_names:
                number_l = rnd_numbers_list()
                genre = i.split('/')[0]
                file_name = os.path.join(hparams.genres_path,i)
                y, sr = readfile(file_name, hparams)
                save_path = os.path.join(hparams.genres_path,genre)
                save_name = i.split('/')[1]
                f.writelines(i + '\n')
                for number in number_l:
                    if number == 0:
                        f.writelines(i.replace('.wav', 'a.wav' + '\n'))
                        data_noise = add_noise(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'a.wav')), data=data_noise,
                                 samplerate=hparams.sample_rate)
                    elif number == 1:
                        f.writelines(i.replace('.wav', 'b.wav' + '\n'))
                        data_roll = shift(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'b.wav')), data=data_roll,
                                 samplerate=hparams.sample_rate)
                    elif number == 2:
                        f.writelines(i.replace('.wav', 'c.wav' + '\n'))
                        data_stretch = stretch(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'c.wav')), data=data_stretch,
                                 samplerate=hparams.sample_rate)
                    elif number == 3:
                        f.writelines(i.replace('.wav', 'd.wav' + '\n'))
                        pitch_speed = change_pitch_and_speed(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'd.wav')), data=pitch_speed,
                                 samplerate=hparams.sample_rate)
                    elif number == 4:
                        f.writelines(i.replace('.wav', 'e.wav' + '\n'))
                        pitch = change_pitch(y, hparams.sample_rate)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'e.wav')), data=pitch,
                                 samplerate=hparams.sample_rate)
                    elif number == 5:
                        f.writelines(i.replace('.wav', 'f.wav' + '\n'))
                        speed = change_speed(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'f.wav')), data=speed,
                                 samplerate=hparams.sample_rate)
                    elif number == 6:
                        f.writelines(i.replace('.wav', 'g.wav' + '\n'))
                        value = value_aug(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'g.wav')), data=value,
                                 samplerate=hparams.sample_rate)
                    elif number == 7:
                        f.writelines(i.replace('.wav', 'h.wav' + '\n'))
                        y_harmonic, y_percussive = hpss(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'h.wav')), data=y_percussive,
                                 samplerate=hparams.sample_rate)
                    elif number == 8:
                        f.writelines(i.replace('.wav', 'i.wav' + '\n'))
                        y_shift = shift(y)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'i.wav')), data=y_shift,
                                 samplerate=hparams.sample_rate)
    print('finished')
if __name__ == '__main__':
    main()