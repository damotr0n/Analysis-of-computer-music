from src import ExtractorWrapper as EW
import subprocess as sp, json, os

class EssentiaWrapper(EW.ExtractorWrapper):

    def __init__(self):
        
        self.features = [
            "acoustic",
            "aggressive",
            "electronic",
            "happy",
            "party",
            "relaxed",
            "sad"
        ]

        ## Two different executables as I work on two different machines
        # self.executable_name = "./essentia_streaming_extractor_music (MAC)"
        self.executable_name = "./essentia_streaming_extractor_music (Ubuntu)"
        self.profile_name = "./profiles/config.yaml"

    def extract_features(self, input_filename:str):

        # call subprocess, create temp file
        # parse temp file and extract necessary data

        sp.call([self.executable_name, input_filename, "temp.txt", self.profile_name])

        feature_list = {}

        with open("temp.txt") as feature_file:
            data = json.load(feature_file)
            for mood in data['highlevel']:

                label = data['highlevel'][mood]['value']
                probability = data['highlevel'][mood]['probability']

                if "not_" in label:
                    label = label[4:]
                    probability = 1 - probability

                feature_list[label] = probability
        
        os.remove("temp.txt")

        return feature_list
