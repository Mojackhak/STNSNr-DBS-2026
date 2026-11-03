"""
# Necessary parameters for anat/lead pipeline
1. cols_basic
2. region_list
3. atlas_name
4. threshold
5. side_dict
6. matfilename
7. save_name
8. subj_meta_file
9. severe_brain_side_col

"""

from plugin.default.tabular_default import cols_basic

region_list = ['SNr', 'STN']
atlas_name = r"Custom_Ewert_Zhang_Middlebrooks"
threshold = 0.05
side_dict = {'lh': 1, 'rh': 0}

subj_meta_file = r'subj_scale.xlsx'

severe_brain_side_col = 'SevereBrainSide'  # column in subj_meta_file indicating side
