from semantic_segmentation import semantic_segmentation

segmenter = semantic_segmentation("S0", "./Data/", save_path="./Models/")

segmenter = semantic_segmentation("./Models/", "./Data/", save_path="./Models/")

segmenter.train(10)

segmenter.predict("./Data/Test", "./Output/")

#segmenter.evaluate()