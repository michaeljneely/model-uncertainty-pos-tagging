'''Define configuration classes'''

class Config:
    '''Defines the parameters used across the code base which can be adjusted without modifying code directly'''
    logger_name = 'tag-uncertainty'
    package_name = 'dl4nlp_pos_tagging'

    # 
    serialization_base_dir = 'outputs'

    # 
    seeds = [87, 2134, 5555]

    #
    mpl_style = 'seaborn-poster'

    #
    sns_palette = 'cubehelix'
