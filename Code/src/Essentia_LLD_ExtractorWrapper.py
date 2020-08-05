from src import ExtractorWrapper as EW
import subprocess as sp, json, os

class EssentiaWrapper(EW.ExtractorWrapper):

    def __init__(self):
        
        self.features = [
            'barkbands_crest', 
            'barkbands_flatness_db', 
            'barkbands_kurtosis', 
            'barkbands_skewness', 
            'barkbands_spread', 
            'dissonance', 
            'erbbands_crest', 
            'erbbands_flatness_db', 
            'erbbands_kurtosis', 
            'erbbands_skewness', 
            'erbbands_spread', 
            'hfc', 
            'loudness_ebu128', 
            'melbands_crest', 
            'melbands_flatness_db', 
            'melbands_kurtosis', 
            'melbands_skewness', 
            'melbands_spread', 
            'pitch_salience', 
            'silence_rate_20dB', 
            'silence_rate_30dB', 
            'silence_rate_60dB', 
            'spectral_centroid', 
            'spectral_complexity', 
            'spectral_decrease', 
            'spectral_energy', 
            'spectral_energyband_high', 
            'spectral_energyband_low', 
            'spectral_energyband_middle_high', 
            'spectral_energyband_middle_low', 
            'spectral_entropy', 
            'spectral_flux', 
            'spectral_kurtosis', 
            'spectral_rms', 
            'spectral_rolloff', 
            'spectral_skewness', 
            'spectral_spread', 
            'spectral_strongpeak', 
            'zerocrossingrate', 
            'barkbands', 
            'erbbands', 
            'gfcc', 
            'melbands', 
            'melbands128', 
            'mfcc', 
            'spectral_contrast_coeffs', 
            'spectral_contrast_valleys'
        ]

        # self.executable_name = "./essentia_streaming_extractor_music (MAC)"
        self.executable_name = "./essentia_streaming_extractor_music (Ubuntu)"
        self.profile_name = "./profiles/music_config.yaml"

    def extract_features(self, input_filename:str):

        # call subprocess, create temp file
        # parse temp file and extract necessary data

        sp.call([self.executable_name, input_filename, "temp.txt", self.profile_name])

        feature_list = {}

        with open("temp.txt") as feature_file:
            data = json.load(feature_file)
            for feature in data['lowlevel']:

                # Skip all features that only have a value
                if type(data['lowlevel'][feature]) is dict:

                    # Special case for one of the features
                    if feature == 'loudness_ebu128':
                        feature_list[feature + '_momentary'] = data['lowlevel'][feature]['momentary']['mean']
                        feature_list[feature + '_short_term'] = data['lowlevel'][feature]['short_term']['mean']
                    # Special case for lists of features instead of singular values
                    elif type(data['lowlevel'][feature]['mean']) is list:
                        data_list = data['lowlevel'][feature]['mean']
                        label_no = 1
                        for point in data_list:
                            feature_list[feature + "_" + str(label_no)] = point
                            label_no = label_no + 1

                    else:
                        feature_list[feature] = data['lowlevel'][feature]['mean']

        os.remove("temp.txt")

        return feature_list
