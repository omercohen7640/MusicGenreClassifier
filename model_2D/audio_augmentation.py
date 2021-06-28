import librosa
import numpy as np
import os
from model_1D.hparams import hparams
import soundfile as sf


def rnd_numbers_list(low,high):
    length = hparams.aug_number
    if length == 8:
        return range(8)
    number = np.random.randint(low,(high+1))
    number_l = [number]
    while len(number_l) < length:
        number = np.random.randint(low,(high+1))
        while number in number_l:
            number = np.random.randint(low,(high+1))
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


def change_pitch_and_speed(data, sr , option = "Low"):
    y_pitch_speed = change_speed(data , option)
    y_pitch_speed = change_pitch(y_pitch_speed , sr , option)
    return y_pitch_speed

def change_speed(data, option = "Low"):
    rate = 1
    if option == "Low":
        rate = 0.95
    elif option == "Medium":
        rate = 0.90
    elif option == "High":
        rate = 0.80
    y_speed = data.copy()
    speed_change = np.random.uniform(low=rate, high=(2-rate))
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def change_pitch(data, sr, option = "Low"):
    if option == "Low":
        pitch_pm = 4
    elif option == "Medium":
        pitch_pm = 8
    elif option == "High":
        pitch_pm = 16
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_change = pitch_pm * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def value_aug(data, option = "Low"):
    if option == "Low":
        strength = 3
    elif option == "Medium":
        strength = 5
    elif option == "High":
        strength = 7
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5, high=strength)
    y_aug = y_aug * dyn_change
    return y_aug


def add_noise(data , option = "Low"):
    if option == "Low":
        strength = 0.010
    elif option == "Medium":
        strength = 0.020
    elif option == "High":
        strength = 0.040
    noise = np.random.randn(len(data))
    data_noise = data + strength * noise
    return data_noise


def hpss(data, option = "Low"):
    if option == "Low":
        rate = 1.0
    elif option == "Medium":
        rate = 3.0
    elif option == "High":
        rate = 5.0

    y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'), margin=(1.0,rate))
    return y_harmonic, y_percussive


def shift(data, option = "Low"):
    if option == "Low":
        strength = 10000
    elif option == "Medium":
        strength = 20000
    elif option == "High":
        strength = 30000
    return np.roll(data, strength)


def stretch(data, option = "Low"):
    rate = 1
    if option == "Low":
        rate = 0.95
    elif option == "Medium":
        rate = 0.90
    elif option == "High":
        rate = 0.80
    input_length = len(data)
    streching = librosa.effects.time_stretch(data, rate)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching


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

def main_reduced( option = "Low",genres = None,extra = False ):
    print('Reduced Augmentation')
    if extra:
        for g in genres:
            genre_path = os.path.join(hparams.genres_path,str(g)+'_extra')
            if not os.path.exists(genre_path):
                os.mkdir(genre_path)

    list_names = ['train_list.txt']
    for list_name in list_names:
        file_names = load_list(list_name, hparams)#train list
        if extra:
            text_file_path = os.path.join(hparams.dataset_path, str(list_name)+"_aug_extra")
        else:
            text_file_path = os.path.join(hparams.dataset_path, str(list_name)+"_aug")
        with open(text_file_path ,'w') as f:
            for i in file_names:
                number_l = rnd_numbers_list(0,7)
                genre = i.split('/')[0]
                if extra and genre not in genres:
                    continue
                file_name = os.path.join(hparams.genres_path,i)
                y, sr = readfile(file_name, hparams)
                if extra:
                    save_path = os.path.join(hparams.genres_path,str(genre)+'_extra')
                else:
                    save_path = os.path.join(hparams.genres_path,genre)
                save_name = i.split('/')[1]
                if not extra:
                    f.writelines(i + '\n')
                print(file_name," - ",number_l)
                if extra:
                    prefix = i.split('/')[0]+'_extra'
                    i = prefix +'/'+ i.split('/')[1]
                else:
                    file_line = i.replace('.wav', 'a.wav' + '\n')
                for number in number_l:
                    if number == 0:
                        f.writelines(i.replace('.wav', 'a.wav' + '\n'))
                        data_noise = add_noise(y,option=option)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'a.wav')), data=data_noise,
                                 samplerate=hparams.sample_rate)
                    elif number == 1:
                        f.writelines(i.replace('.wav', 'b.wav' + '\n'))
                        data_roll = shift(y,option=option)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'b.wav')), data=data_roll,
                                 samplerate=hparams.sample_rate)
                    elif number == 2:
                        f.writelines(i.replace('.wav', 'c.wav' + '\n'))
                        data_stretch = stretch(y,option=option)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'c.wav')), data=data_stretch,
                                 samplerate=hparams.sample_rate)
                    elif number == 3:
                        f.writelines(i.replace('.wav', 'd.wav' + '\n'))
                        pitch_speed = change_pitch_and_speed(y,hparams.sample_rate,option=option)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'd.wav')), data=pitch_speed,
                                 samplerate=hparams.sample_rate)
                    elif number == 4:
                        f.writelines(i.replace('.wav', 'e.wav' + '\n'))
                        pitch = change_pitch(y, hparams.sample_rate)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'e.wav')), data=pitch,
                                 samplerate=hparams.sample_rate)
                    elif number == 5:
                        f.writelines(i.replace('.wav', 'f.wav' + '\n'))
                        speed = change_speed(y,option=option)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'f.wav')), data=speed,
                                 samplerate=hparams.sample_rate)
                    elif number == 6:
                        f.writelines(i.replace('.wav', 'g.wav' + '\n'))
                        value = value_aug(y,option=option)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'g.wav')), data=value,
                                 samplerate=hparams.sample_rate)
                    elif number == 7:
                        f.writelines(i.replace('.wav', 'h.wav' + '\n'))
                        y_harmonic, y_percussive = hpss(y,option=option)
                        sf.write(file=os.path.join(save_path, save_name.replace('.wav', 'h.wav')), data=y_percussive,
                                 samplerate=hparams.sample_rate)
    print('finished')


if __name__ == '__main__':
    main()