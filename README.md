# A Smart Precision Irrigation Application for Tomato Vines

In our project, we explored precision irrigation in tomato vines. We can predict the watering needs of tomato vines by analyzing the Crop Water Stress Index (CWSI) of a plant. This is done through visual and thermal imagery with photos captured by a mobile device and uses convolutional neural networks.

Additionally, we investigated different state-of-the-art semantic segmentation neural networks and evaluated their performances on our tomato dataset. Utilizing SegNets to identify sunlit tomoato leaves, we achieved a test accuracy of 82%. 

# Motivation

California farmers have dealt with droughts within the recent years, relying on federal and state supplies to irrigate their crops. Historically, farmers have been skeptical in transitioning to science-based irrigation methods because of its unproven reliability, cost-inefficiency, reliance on user input, and challenging upkeep. Our work done in this project will allow farmers to irrigate their crops based on a science-based method with great accuracy and does not require expensive instruments.

![Thermal Image](/images/thermal.png)
![Visual Image](/images/visual.png)

# Contribution

We capture 1,110 thermal images of tomato vines. With these images, we processed and extracted metadata that is needed to calculate Crop Water Stress Index as well as other attributes. We then manually annotated 1,110 visual images for sunlit leaf semantic segmentation. Using an open-software called Pynovisao, we can customize unsupervised clustering algorithms to support human annotations.

We then trained, evaluated, and testerd 10 state-of-the-art, published, neural networks that are used for semantic segmentation. We graphed four of the top-performing models using commonly used metrics in evaluating the performance of semantic segmentations. These metrics include: accuracy, precision, recall, f1 score, and intersection over union.

![Table](/images/table.png)
![Graph](/images/graph.png)

# Reference
https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
