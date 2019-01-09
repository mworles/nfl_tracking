import pandas as pd
import os
os.chdir('C:/Users/mworley/nflbdb/Data/')

def get_rushes(file):

    print file

    # import file
    gm = pd.read_csv('raw/' + file)

    # some cleaning on tracking file
    gm['event'].fillna('NA', inplace=True)
    #gm = gm[gm['displayName'] != 'football']

    # select all plays from one game
    gid = filter(str.isdigit, file)
    gm_plays = plays[plays['gameId'] == int(gid)]

    # create list of unique playids where playDescription includes string 'handoff'
    ho_events = gm[gm['event'].str.contains('handoff')]
    ho_playids = pd.unique(ho_events['playId']).tolist()

    # use playid list to subset tracking rows and plays
    ho_plays = gm_plays[gm_plays['playId'].isin(ho_playids)]

    # filter out plays that contain one of several strings
    # includes non-designed runs, 2-point attempts, and passes
    rushes = ho_plays[~ho_plays['playDescription'].str.contains('pass')]
    rushes = rushes[~rushes['playDescription'].str.contains('TWO-POINT CONVERSION ATTEMPT')]
    rushes = rushes[~rushes['playDescription'].str.contains('scrambles')]
    rushes = rushes[~rushes['playDescription'].str.contains('Aborted')]

    # create list of tag words for playDescription
    kw = ['left', 'right', 'up', 'rushes', 'scrambles']

    # empty list for ball carrier text from playDescription
    bc_list = []

    # for each playDescription in rushes dataframe
    for x in rushes.loc[:, 'playDescription']:
        # identify the tag words in each play description
        dlist = x.split(' ')
        cmn = list(set(kw).intersection(dlist))
        # if only one tag word, set to tag variable
        if len(cmn) == 1:
            tag = cmn[0]
            tagin = dlist.index(tag)

            # set bc variable to ball carrier text, append to bc_list
            bc = dlist[tagin-1]
            bc_list.append(bc)
        else:
            print 'more than one tag word'
            print x
            bc_list.append('N.invalid')
            pass

    # create list of unique ball carrier names
    bc_set = list(set(bc_list))

    print '%d rushes' % (len(bc_list))
    print bc_set

    # create 2-column df of all unique player id and displayName in game
    gm_players = gm[['nflId', 'displayName']].drop_duplicates()

    # empty dictionary to store ball carrier text and nflId
    bc_dict = {}

    for b in bc_set:

        # get details of ball carrier
        bc_last = b.split('.')[-1]
        bc_first = b.split('.')[0]
        if len(bc_first) > 1:
            bc_first = bc_first[0]

        # get all names from gm_players that include text in bc_last
        potmat = gm_players[gm_players['displayName'].str.contains(bc_last)]

        # if only one match, set ball carrier dict key value pair
        # key is ball carrier text, value is nflId
        if potmat.shape[0] == 1:
            bl = potmat.values.tolist()
            bc_dict[b] = bl[0][0]
        elif potmat.shape[0] == 0:
            bc_dict[b] = 000000
        else:
            print "more than one player has %s" % (bc_last)
            print b
            print bc_first
            print potmat.values
            for p in potmat.values:
                fi = p[1][0]
                if fi == bc_first:
                    bc_dict[b] = p[0]
                    print bc_dict[b]
                    break
                else:
                    pass

    # create list of nflId for each ball carrier text in rushes
    ball_carrier_ids = [bc_dict[x] for x in bc_list]

    # create column in rushes containing nflId for ball carrier
    rushes.loc[:, 'nflId'] = ball_carrier_ids

    # inner join to filter tracking data on ball carrier nflId
    df = pd.merge(gm, rushes, on=['gameId', 'playId', 'nflId'], how='inner')

    # inner join with ball position
    ball = gm[gm['displayName'] == 'football']
    ball = ball.loc[:, ['gameId', 'playId', 'frame.id', 'x', 'y', 's', 'dis', 'dir']]
    ball.rename(columns={'x': 'x_ball',
                         'y': 'y_ball',
                         's': 's_ball',
                         'dis': 'dis_ball',
                         'dir': 'dir_ball'}, inplace=True)
    df = pd.merge(df, ball, on=['gameId', 'playId', 'frame.id'])

    return df

# import data for all plays
plays = pd.read_csv('raw/plays.csv')
files = [f for f in os.listdir('raw/') if 'tracking_gameId_' in f]

gm_rushes = map(lambda x: get_rushes(x), files)
df = pd.concat(gm_rushes)
df.to_csv('interim/rushes.csv')
