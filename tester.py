from semantic_segmentation import semantic_segmentation

segmenter = semantic_segmentation("./Models/")
segmenter.evaluate("./Data/Validation/")