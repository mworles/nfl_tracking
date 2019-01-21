import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from speed_extractor import extract_speed_features, filter_speed_features
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    #extract_speed_features('rush_toLOS')
    #extract_speed_features('rush_tocontact')
    #extract_speed_features('rush_contact')
