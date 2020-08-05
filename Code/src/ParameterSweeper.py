from src import SynthWrapper as SW
from src import ExtractorWrapper as EW
import math, os, json

class ParameterSweeper():

    def __init__(self, synthIn:SW.SynthWrapper, extractorIn:EW.ExtractorWrapper):
        self.synth = synthIn
        self.extractor = extractorIn

        self.filename_no = 0

        self.export_data = {}
        self.export_data['data'] = []
    

    def sweep_parameters(self, parameter_num = 0, parameter_list = []):
        # set current parameters
        # if next one exists, pass list onto that one
        # otherwise generate sound

        curr_param = self.synth.parameters[parameter_num]

        # Find number of steps based on range and increment
        steps = math.floor((curr_param["max"] - curr_param["min"]) / curr_param["increment"])

        for p in range(steps + 1):

            param = curr_param["min"] + p * curr_param["increment"]
            if len(parameter_list) < len(self.synth.parameters):
                parameter_list.insert(parameter_num, param)
            else:
                parameter_list[parameter_num] = param

            if parameter_num < len(self.synth.parameters) - 1:
                self.sweep_parameters(parameter_num + 1, parameter_list)
            else:
                self.filename_no += 1
                filename = 'sound' + str(self.filename_no)
                if self.synth.generate_sound(filename, parameter_list):

                    data = {}
                    data['Parameters'] = {}

                    for param_no in range(len(parameter_list)):
                        data['Parameters'][self.synth.parameters[param_no]['name']] = parameter_list[param_no]

                    features = self.extractor.extract_features(filename + '.wav')
                    os.remove(filename + '.wav')
                    data['Features'] = features

                    self.export_data['data'].append(data)

        # with open('data.json', 'w') as outfile:
        #     json.dump(self.export_data, outfile, indent=2)
    
    def dump_data(self):
        with open('data.json', 'w') as outfile:
            json.dump(self.export_data, outfile, indent=2)