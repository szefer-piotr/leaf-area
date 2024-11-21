# class LeafFrameDataset(Dataset):
#     def __init__(self, data, transformations, target_name = 'area_lost', 
#                  ):
#         """
#         Arguments:
#             dataset (string): Path to the csv file with tabular data.
#             transformations (list): List of transformations on the images
#             target_name (string): Name of the target column.
#         """
#         self.data = data
#         self.target_name = target_name
#         self.transformations = transformations

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
        
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_path = self.data.iloc[idx]['image_path']
#         # img = cv2.imread(img_path)
#         img = Image.open(img_path)
        
#         if self.transformations is not None:
#             img = self.transformations(img)
        
#         target = data.iloc[idx][self.target_name].astype('float32')
#         # sample = {'image': img, 'target': target}
        
#         return img, target