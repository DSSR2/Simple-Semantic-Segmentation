from semantic_segmentation import semantic_segmentation

segmenter = semantic_segmentation("./Data/", "S3", save_path="./Models/")
segmenter.train(10)
