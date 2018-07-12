from dataset import DataLoader, LabeledPathListingGenerator, DataSet

if __name__ == '__main__':
    data_loader = DataLoader("video_label.pkl", "label_encoding.pkl")
    train_set, val_set, test_set = data_loader.generate_split([0.7, 0.2, 0.1])
    generator = train_set.get_batch_generator(1, 160, 120, 15)
    print(next(generator))