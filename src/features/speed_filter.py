import sys
sys.path.insert(0, 'c:/users/mworley/nfl_tracking/src/features')
from speed_extractor import extract_speed_features, filter_speed_features
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    filter_speed_features('rush_toLOS', 'success')
    filter_speed_features('rush_tocontact', 'success')
    filter_speed_features('rush_contact', 'success')
    filter_speed_features('rush_toLOS', 'yards')
    filter_speed_features('rush_tocontact', 'yards')
    filter_speed_features('rush_contact', 'yards')
    filter_speed_features('rush_toLOS', 'EPA')
    filter_speed_features('rush_tocontact', 'EPA')
    filter_speed_features('rush_contact', 'EPA')
    filter_speed_features('rush_tocontact', 'yards_aftercont')
    filter_speed_features('rush_contact', 'yards_aftercont')
    filter_speed_features('rush_toLOS', 'yards_aftercont')
