# **MUSA 6950-001 202510 AI for Urban Sustainability**

## **Flood Mapping for Urban Disruptions and Emergency Response**

Shuya Guan

## **Objectives**
As the frequency and severity of extreme weather events escalate due to climate change, urban transportation infrastructure—including highways, rail systems, and bridges—faces increasing exposure to flood-related disruptions. Recent analyses reveal that the global population residing in flood-prone areas has increased by 20% to 24% since the year 2000, a rate significantly higher than previously projected, with climate-related drivers such as intensified rainfall and sea-level rise cited as major contributors (Environmental Defense Fund, n.d.).
Despite growing awareness, existing flood risk assessments often lack the spatial resolution and predictive capability required by transportation planners to effectively prioritize adaptation strategies. This study proposes the use of machine learning techniques to classify satellite imagery into flooded and non-flooded categories. The principal objective is to enhance flood detection accuracy using Sentinel satellite data, thereby supporting timely disaster response and advancing climate-resilient transportation planning.

<br>

## **Problem Scope**
This research aims to identify flood-prone areas and facilitate a more targeted approach to risk mitigation. Our analysis that can be overlaid with critical transportation corridors, to highlight infrastructure segments that are most at risk. These insights can be used to inform decisions on infrastructure investment, emergency detour planning, and long-term design adaptations to climate hazards.
The urgency of this work is underscored by recent hydrological models projecting a global increase in flooding of 9% to 49% by the end of the 21st century under various climate scenarios (Fathom, 2024). Additionally, compounding factors such as rapid urbanization, outdated drainage systems, and land subsidence are contributing to more frequent and severe flood events (CAF America, 2024). These dynamics necessitate the development of data-driven decision support systems to augment traditional flood management frameworks.

<br>

## **Target Users**
The outputs of this project are intended to serve a range of stakeholders engaged in urban infrastructure and emergency management, particularly those focused on the transportation sector. Key user groups include:
Urban and Regional Planners with a focus in integrating flood vulnerablity into long-range infrastructure planning and zoning decisions.
Public Transit Agencies with a need for real-time operational decisions such as route suspensions or diversions during flood events.
Civil and Transportation Engineers who incorporate flood risk parameters into the design and retrofitting of transportation infrastructure.
By equipping these actors with a predictive tools for rapid flood detection, the study aims to improve the resilience and adaptability of transportation systems in the face of intensifying climate risks.

# 1. Dataset Exploration

**1.1 Data Source Introduction:**

The dataset used in this study is the SEN12-FLOOD datase (available through its DOI: 10.21227/w6xz-s898.), which consists of 336 time series collected from regions affected by major flooding events during the winter of 2019. Each time series includes both Sentinel-1 (SAR grayscale) and Sentinel-2 (RGB optical) satellite images, covering flooded and non-flooded scenes. The dataset is particularly valuable for flood detection tasks because it provides multi-modal imagery with both active and passive sensing data.

The spatial coverage includes areas clustered in East Africa, South West Africa, the Middle East, and Australia, representing diverse geographic and climatic conditions. In total, the dataset contains over 28,000 labeled images, with 17,961 flooded and 10,333 non-flooded samples. Additionally, Sentinel-2 accounts for a larger portion of the data folders compared to Sentinel-1, providing a richer set of RGB imagery suitable for deep learning models.

This combination of multi-source satellite imagery and clear flood labels makes SEN12-FLOOD a robust benchmark for training and evaluating flood classification models.

**1.2 Dataset Description**

This study uses the SEN12-FLOOD dataset, a multi-source satellite image collection designed for flood detection tasks. The dataset comprises 336 time series of both Sentinel-1 (SAR grayscale) and Sentinel-2 (RGB optical) images, covering regions affected by major flooding events during the winter of 2019. These time series are organized in numbered folders (e.g., 0005, 0063) and contain labeled instances of flooded and non-flooded areas, based on binary labels extracted from metadata (where 1 indicates flooded and 0 indicates non-flooded).

A script-based scan of the dataset directory confirmed the presence of 28,300 TIFF images, including Sentinel-1 radar data with VV and VH polarizations, as well as Sentinel-2 optical imagery. According to the label distribution, the dataset includes 17,967 flooded images (approximately 63%) and 10,333 non-flooded images (approximately 37%), offering a reasonably balanced and well-labeled resource for supervised learning.

Visual inspection of sample images shows that flooded areas typically exhibit more defined water patterns and distinct textures, while non-flooded areas present more stable land features. Some Sentinel-1 images appear very dark and may require preprocessing to enhance signal quality and texture visibility, especially when used in combination with optical data.
**1.3 Sentinel-2 Selection for CNN Modeling**

Given the task’s reliance on RGB imagery for deep learning, particularly convolutional neural networks (CNNs), I focus on using Sentinel-2 data for model development. This decision is supported by several factors:

RGB Composition: Sentinel-2 provides bands B04 (red), B03 (green), and B02 (blue), which together form true-color RGB images ideal for training CNN-based image classification or segmentation models.

Spatial Resolution: The RGB bands of Sentinel-2 have a 10-meter pixel resolution, providing clearer visual information than Sentinel-1 grayscale radar images. This allows better distinction of features like roads, water boundaries, and vegetation textures.

Data Quantity: The dataset contains over 24,000 Sentinel-2 images, offering a large and diverse input space that supports robust model training and evaluation.

In contrast, Sentinel-1 provides only single-channel grayscale data (VV and VH), which lacks the spectral and color richness needed for CNNs that depend on RGB input. For this reason, Sentinel-2 is prioritized for vision-based modeling in this study.

# 2. Data Preprocessing Pipeline
**2.1 Data Preprocessing**

To ensure model reliability and input consistency, I applied a structured preprocessing pipeline focused on clean data selection, RGB band alignment, and subset-specific transformations.

**a.Band Selection and Filtering**
Each sample folder in the SEN12-FLOOD dataset contains multiple Sentinel-2 .tif images, each corresponding to a specific spectral band and acquisition date. To construct valid RGB images, I required the presence of B02 (blue), B03 (green), and B04 (red) bands captured on the same date. This step is critical, as Sentinel-2 captures images on different dates for different bands, and mismatched timestamps can lead to corrupted RGB compositions.

A custom script was implemented to iterate through all 335 folders, identify valid band-date combinations, and retain only those folders where all three required bands were available for at least one shared timestamp. This resulted in 300 valid folders, confirming that ~90% of the dataset is suitable for RGB reconstruction.

**b. Reflectance Scaling and Normalization**
For RGB generation, raw band values were read using rasterio and processed as float32 arrays. To enhance feature contrast while preserving dynamic range, I applied percentile-based normalization: band values were stretched between their 2nd and 98th percentiles, clipped, and scaled to [0, 1] before being converted to 8-bit RGB images. This method helps minimize the impact of outliers while maintaining visual clarity. Alternatives such as dividing by a constant (e.g., 10,000 for reflectance scaling) were considered, following ESA recommendations, but percentile scaling offered better empirical contrast.

**c. Cloud Filtering**
During manual review, several images were found to be dominated by heavy cloud coverage or dark/noisy regions. These cases were implicitly filtered out by the valid-folder check, since folders lacking usable RGB bands on the same date were excluded. While more advanced cloud detection could be added, this simple filter was effective in improving overall quality.

**d. Dataset Splitting**
After filtering, the 300 valid folders were split by folder into training (70%, 210 folders), validation (15%, 45 folders), and test (15%, 45 folders) sets. This folder-level splitting ensures no spatial or temporal leakage between subsets.

**e. Transformations**
To preserve data integrity and allow subset-specific handling, transformations were applied after splitting:

*Training set:* images were resized to 224×224, then augmented with random horizontal flips and small-angle rotations to increase robustness. Normalization to a range of [–1, 1] was also applied via (x – 0.5) / 0.5.

*Validation and test sets:* only resizing and normalization were applied, keeping them clean for unbiased evaluation.

I aim to use this approach allowing the training process to benefit from diversity and regularization via augmentation, while ensuring that validation and test sets accurately reflect real-world input distributions.

**2.2 Dataset Construction and Splitting Strategy**

In this project, each folder in the SEN12-FLOOD dataset is treated as one scene—meaning a specific location at a specific time. A valid scene contains three Sentinel-2 bands (B02, B03, and B04) from the same date, which I use to form an RGB image. Each folder also comes with a binary label: 1 for flooded, 0 for non-flooded.

To avoid data leakage, I split the dataset by folder rather than by image. This is important because nearby or repeated scenes can look very similar. Random image-level splitting could put almost identical samples in both training and test sets, leading to misleading results. I followed best practice by shuffling all 300 valid folders and splitting them into 70% for training (210 folders), 15% for validation (45), and 15% for testing (45), making sure each set contains distinct scenes.

I used a custom PyTorch Dataset class to load RGB images and labels. For training, we applied resizing, normalization, and light augmentation (random flip and rotation). For validation and testing, I only used resizing and normalization to keep evaluation consistent.

If a folder fails to load properly, the dataset class skips it and tries a nearby one. This makes the dataloader more stable. Each sample becomes a 3-channel tensor of shape [3, 224, 224], with a binary label. A sample batch confirmed that the pipeline runs as expected.

# 3. Model Development
Before building our own model, I looked at two existing approaches that are commonly used for flood detection: UN-SPIDER and WorldFloods. They take very different approaches and help highlight the trade-offs involved in flood mapping with satellite data.

*UN-SPIDER*, developed by the United Nations, is a rule-based method that relies on comparing pre- and post-event Sentinel-1 SAR images. It uses simple thresholds—typically based on changes in backscatter values—to detect flooded areas. Sometimes, it includes additional layers like elevation or land cover, or uses basic classifiers like random forests if training data is available. The main advantage is that it’s lightweight and easy to run, even with limited computing resources. But the downside is its low accuracy and inability to capture complex features, especially in urban areas.

*WorldFloods*, developed by Oxford and the Frontier Development Lab, uses a U-Net deep learning model trained on multi-temporal Sentinel-1 SAR inputs. It produces pixel-wise flood maps and performs well, even under cloud cover. However, it requires multi-source inputs, including several SAR bands and often elevation data. This makes it harder to set up and less flexible in situations where only RGB or limited data is available.

These two models show a clear trade-off: one is simple but limited, the other is powerful but complex. The goal is to explore a middle ground—using only RGB images from Sentinel-2 to build a model that is easier to train and deploy, while still offering reasonable accuracy. This helps lower the barrier to entry and increases the model’s applicability in real-world settings where access to multi-modal data or high-end hardware may be limited.

# 3.1 Random Forest Baseline
As an initial baseline, I trained a *Random Forest* classifier using flattened Sentinel-2 RGB images. Since *Random Forests* require tabular input, I first converted each 3×224×224 image into a flat vector and stored the entire dataset as NumPy arrays. This process allowed us to feed the data into the model, though it came at the cost of losing spatial structure.

I used 200 estimators, limited the tree depth to 15, and applied class weighting to handle the slight imbalance between flooded and non-flooded samples. Cross-validation on the training set produced an average accuracy of 66.2%, with some variation across folds. On the validation set, we saw an overall accuracy of 60%. While the model performed well at detecting flooded scenes (recall of 0.88), it struggled with non-flooded areas, where recall dropped to 0.21. The confusion matrix clearly showed this skew, with many non-flooded samples being misclassified as flooded.

Still, the ROC curve gave us an AUC of 0.705, suggesting the model was learning a reasonably good separation, even if thresholding was off. These results highlight the limits of traditional models like *Random Forest* when applied to high-dimensional image data. Flattening the images removed spatial relationships, which are often crucial for visual tasks like flood detection. This baseline helped clarify the need for a model that can directly work with image structure—something we address with our CNN-based approach.

# 3.2 CNN-Based Model: ResNet18
To take advantage of the spatial structure in satellite images, I trained a convolutional neural network using a ResNet18 backbone. I modified the final fully connected layer for binary classification and used cross-entropy loss with the Adam optimizer. The model was trained on RGB images of shape 3×224×224, and we applied early stopping based on validation loss to avoid overfitting.

From the start, the model showed strong performance on the training set, reaching over 95% accuracy within a few epochs. On the validation set, performance fluctuated more, but eventually stabilized around 82%. Although validation loss did not decrease consistently, the accuracy and ROC curve suggest that the model was learning useful patterns. Our best validation AUC reached 0.926, showing a clear separation between classes.

Looking at the confusion matrix, the model correctly predicted 15 of 19 non-flooded samples and 22 of 26 flooded ones. Precision and recall were balanced across both classes, each around 0.79–0.85, leading to an F1 score of 0.846 overall. Compared to the Random Forest baseline, the ResNet model handled both classes more evenly and achieved a much higher AUC, suggesting that the convolutional architecture was better able to capture local textures and spatial context that are critical in flood detection. Overall, the ResNet18 model provided a clear improvement in performance, especially in its ability to generalize across classes and to preserve meaningful patterns in the imagery. While the training set was small, the model's results were stable, and the use of RGB alone—without any additional SAR or elevation data—demonstrated strong potential for lightweight flood mapping tasks.

# 3.3 EfficientNet-B0 Model
To further explore the potential of CNNs in RGB-based flood detection, I fine-tuned an EfficientNet-B0 model pre-trained on ImageNet. I replaced the final classification layer with a two-class output and trained the model using AdamW with a learning rate of 1e-4. Early stopping was applied based on validation loss, with a patience of 3 epochs.

Training progressed smoothly, and the model reached high accuracy quickly. After about five epochs, training accuracy stabilized above 97%, and validation accuracy hovered consistently around 88–89%. Compared to the previous ResNet18 model, EfficientNet converged faster and showed slightly more stable validation performance. The validation loss also improved steadily, reaching its minimum near epoch 11.

Performance on the validation set was strong across all metrics. The overall accuracy reached 88.9%, with an F1 score of 0.902. Both classes were handled well: the model correctly identified 17 of 19 non-flooded areas and 23 of 26 flooded areas. Precision and recall were well balanced, as confirmed by the confusion matrix.

The ROC curve further highlighted the model’s discriminative power, with an AUC of 0.9706. This is the highest among all models tested, suggesting that EfficientNet was particularly effective at learning subtle visual patterns from RGB imagery.

EfficientNet-B0 not only matched but slightly outperformed the ResNet baseline, especially in terms of calibration and class balance. Its lightweight architecture and strong validation results suggest it’s a solid choice for flood detection tasks where both accuracy and efficiency matter.

# 4. Model Comparison
After training and evaluating all three models on the held-out test set, we observed clear differences in performance between traditional and deep learning approaches. The Random Forest baseline struggled to generalize, achieving only 46% accuracy and an AUC of 0.56. Its predictions were skewed, favoring the flooded class while misclassifying most non-flooded scenes. This aligns with the earlier observation that flattening RGB images discards spatial structure, which is essential in geospatial tasks. The ResNet model showed a significant improvement, reaching 71% accuracy and a balanced F1 score of 0.71. However, it occasionally confused similar-looking scenes across classes.

EfficientNet-B0 performed best overall: even after being retrained on the combined train+val set, it maintained robust accuracy (77.8%) and the highest AUC of 0.92, suggesting stable decision boundaries and better confidence calibration. Both classes were handled evenly, with precision and recall above 0.74. Visualizing misclassified samples also helped us understand edge cases. Most errors occurred in scenes with partial flooding, dense cloud cover, or ambiguous terrain. While EfficientNet did not completely eliminate misclassifications, it consistently outperformed the other models and generalized well across geographic and visual variation in the test set.

# 5. Final Model
Based on comparative validation results,  I selected EfficientNet-B0 as the final model for this task. It consistently outperformed other models during validation and demonstrated the best balance of accuracy and robustness.

To fully leverage the available data, I combined the original training and validation sets for final training. A small portion was still held out for early stopping to avoid overfitting. The model was trained using this extended dataset, and the best-performing checkpoint (based on validation loss) was saved. I then evaluated the final model on the independent test set, which had not been seen during training or validation.

On the test set, EfficientNet-B0 achieved an accuracy of 77.8%, an F1 score of 0.78, and a strong AUC of 0.92. These results indicate that the model maintained stable confidence and good discrimination between the flooded and non-flooded classes. Both classes were predicted with balanced precision and recall (ranging from 0.75 to 0.82), confirming the model’s generalizability. Although a few misclassifications remained—mostly involving ambiguous terrain or cloud-covered regions—the model's overall performance significantly surpassed that of Random Forest and ResNet, justifying its selection for final deployment or reporting.

# 6. Conclusion
This project set out to evaluate how different machine learning models perform in classifying flood-affected areas using only Sentinel-2 RGB imagery. Starting with a simple Random Forest baseline and progressing through deep learning models like ResNet18 and EfficientNet-B0, we gradually refined our approach toward more spatially aware and robust solutions.

Among all models tested, EfficientNet-B0 achieved the best balance between accuracy, F1 score, and AUC, confirming the value of deeper convolutional architectures in extracting high-level spatial patterns from remote sensing data. Its consistent results across different samples demonstrated that RGB data alone, when properly preprocessed and modeled, can be surprisingly effective in flood detection tasks.

Beyond model performance, the project also represents a complete pipeline: from raw TIFF image handling, preprocessing and dataset curation, to evaluation and visualization. It provided a comprehensive view of what it takes to apply modern vision techniques to satellite imagery in a real-world scenario. The workflow and code we developed can now be extended to future projects involving multi-modal or time-series inputs.

This work reinforces the potential of deep learning in disaster monitoring and builds a solid foundation for scaling toward more complex Earth observation tasks.

# 7. References
Mobley, W., Sebastian, A., Blessing, R., Highfield, W. E., Stearns, L., & Brody, S. D. (2021). Quantification of continuous flood hazard using random forest classification and flood insurance claims at large spatial scales: A pilot study in southeast Texas. Natural Hazards and Earth System Sciences, 21(2), 807–822. https://doi.org/10.5194/nhess-21-807-2021​


Roy, S. (n.d.). Flood prediction models performance comparison [Kaggle Notebook]. Kaggle. https://www.kaggle.com/code/subhojeetroy01/flood-prediction-models-performance-comparison​


DanishKaggle78. (n.d.). Flood detection ML project [Kaggle Notebook]. Kaggle. https://www.kaggle.com/code/danishkaggle78/flood-detection-ml-project/notebook

CAF America. (2024, September 25). Global flooding crisis 2024: Causes, impact, and relief efforts. https://cafamerica.org/blog/global-flooding-crisis-2024-causes-impact-and-relief-efforts/


Environmental Defense Fund. (n.d.). Why are floods hitting more places and people? https://www.edf.org/why-are-floods-hitting-more-places-and-people


Fathom. (2024, August 21). Global flooding to increase 9%-49% this century, new study reveals. https://www.fathom.global/newsroom/new-study-reveals-global-flooding-increase/


Brown, P. (2024, May 8). Are floods dramatically increasing due to climate change? The Breakthrough Institute. https://thebreakthrough.org/journal/no-20-spring-2024/are-floods-dramatically-increasing-due-to-climate-change


