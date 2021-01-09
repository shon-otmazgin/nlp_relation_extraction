# import pandas as pd
# import sys
#
# from utils import WORK_FOR, read_annotations
#
# df = pd.read_pickle(sys.argv[1])
# print(f'dataframe size from pickle: {df.shape}')
# print(f'response varibale (y) counts:\n{df["y"].value_counts()}')
#
# annotations = read_annotations(sys.argv[2])
# c = 0
# for sent_id, annots in annotations.items():
#     for a in annots:
#         if a[1] == WORK_FOR:
#             # print(f'{sent_id} {ann}')
#             c+=1
# print(f'{WORK_FOR} input annotations: {c}')
#
# print()
# c = 0
# for i, idx in enumerate(df[df['y'] == 0].index):
#     sent_id, arg1, arg2 = idx
#     work_for_annots = [ann for ann in annotations[sent_id] if ann[1] == WORK_FOR]
#     if not work_for_annots:
#         continue
#
#     print(sent_id)
#     print(f'annotations:')
#     print(work_for_annots)
#
#     res = input(f'{i}. {arg1} {WORK_FOR} {arg2}? (y/n): ')
#     print()
#     if res == 'y':
#         df.loc[idx, 'y'] = 1
#
# print(df['y'].value_counts())
#
# print(f'dataframe size to pickle: {df.shape}')
# df.to_pickle("train_df.pkl")
# print(f'dataframe saved as: train_df.pkl')